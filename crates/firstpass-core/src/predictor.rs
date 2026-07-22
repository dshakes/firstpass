//! Per-query gate-pass predictor (ADR 0008, Phase 2).
//!
//! A hand-rolled **online logistic regression** that estimates `P(gate-pass | rung, query
//! features)` — the per-query, per-rung success probability the start-rung bandit's coarse
//! context buckets can't express. It is trained incrementally from the deployment's own
//! receipts (each attempt is a labeled example) and, in this phase, its prediction is recorded
//! on the receipt in **shadow** — it never changes routing. Whether the prediction is good
//! enough to *act* on is decided later, offline, via `firstpass predictor-eval` (AUC / Brier).
//!
//! No ML/linalg dependency — the model is a weight vector and two closed-form updates. The
//! feature encoding is fixed-length and deterministic so predictions are reproducible and the
//! receipt is stable.

use crate::features::{Features, TaskKind};

/// Number of `TaskKind` variants (one-hot width). Keep in sync with [`TaskKind`].
const N_TASK_KINDS: usize = 7;
/// Rungs one-hot width (rungs beyond this fold into the last slot).
const N_RUNGS: usize = 8;
/// Fixed feature-vector length. Layout (see [`encode`]):
/// `[bias, task_kind×7, prompt_bucket_norm, has_tools, tool_count_norm, has_images,
///   session_fail_norm, rung×8]` = 1 + 7 + 1 + 1 + 1 + 1 + 1 + 8 = 21.
pub const FEATURE_DIM: usize = 1 + N_TASK_KINDS + 5 + N_RUNGS;

/// Index of the (unregularized) bias term.
const BIAS: usize = 0;

/// One-hot slot for a task kind, in the block that starts right after the bias.
fn task_kind_slot(k: TaskKind) -> usize {
    let i = match k {
        TaskKind::CodeEdit => 0,
        TaskKind::TestGen => 1,
        TaskKind::Explore => 2,
        TaskKind::Review => 3,
        TaskKind::Extract => 4,
        TaskKind::Chat => 5,
        TaskKind::Other => 6,
    };
    1 + i
}

/// Numerically safe logistic sigmoid.
fn sigmoid(z: f64) -> f64 {
    let z = z.clamp(-30.0, 30.0);
    1.0 / (1.0 + (-z).exp())
}

/// Encode `(features, rung)` into the fixed-length feature vector (see [`FEATURE_DIM`]).
///
/// Deterministic and privacy-preserving — it reads only the already-coarsened [`Features`]
/// (buckets and flags, never raw prompt text).
#[must_use]
pub fn encode(features: &Features, rung: u32) -> [f64; FEATURE_DIM] {
    let mut x = [0.0_f64; FEATURE_DIM];
    x[BIAS] = 1.0;
    x[task_kind_slot(features.task_kind)] = 1.0;
    let base = 1 + N_TASK_KINDS;
    // prompt token bucket is already floor(log2 tokens); normalize by a generous ceiling.
    x[base] = f64::from(features.prompt_token_bucket).min(16.0) / 16.0;
    x[base + 1] = f64::from(u8::from(features.tool_count > 0));
    x[base + 2] = f64::from(features.tool_count).min(10.0) / 10.0;
    x[base + 3] = f64::from(u8::from(features.has_images));
    x[base + 4] = f64::from(features.session_failure_count).min(5.0) / 5.0;
    let rung_slot = 1 + N_TASK_KINDS + 5 + (rung as usize).min(N_RUNGS - 1);
    x[rung_slot] = 1.0;
    x
}

/// An online logistic-regression predictor of gate-pass probability.
///
/// Cheap to update (one SGD step per observed attempt) and cheap to query (one dot product).
/// Wrap in `Arc<Mutex<_>>` for the proxy, like the bandit — per-process, in-memory; the
/// durable state is the receipts it warm-starts from.
#[derive(Debug, Clone)]
pub struct PassPredictor {
    weights: [f64; FEATURE_DIM],
    lr: f64,
    l2: f64,
}

impl PassPredictor {
    /// Create a predictor with learning rate `lr` (0, 1] and L2 penalty `l2` (>= 0), zero-init.
    #[must_use]
    pub fn new(lr: f64, l2: f64) -> Self {
        Self {
            weights: [0.0; FEATURE_DIM],
            lr,
            l2,
        }
    }

    /// Predicted `P(gate-pass)` for `(features, rung)`, in `(0, 1)`.
    #[must_use]
    pub fn predict(&self, features: &Features, rung: u32) -> f64 {
        let x = encode(features, rung);
        let z: f64 = self
            .weights
            .iter()
            .zip(x.iter())
            .map(|(w, xi)| w * xi)
            .sum();
        sigmoid(z)
    }

    /// One SGD step against the observed outcome `passed` for `(features, rung)`.
    ///
    /// Gradient of the logistic loss: `(p - y)·x`, plus L2 shrinkage on every weight **except**
    /// the bias (regularizing the intercept would bias the base rate).
    pub fn update(&mut self, features: &Features, rung: u32, passed: bool) {
        let x = encode(features, rung);
        let p = {
            let z: f64 = self
                .weights
                .iter()
                .zip(x.iter())
                .map(|(w, xi)| w * xi)
                .sum();
            sigmoid(z)
        };
        let err = p - f64::from(u8::from(passed));
        for (i, w) in self.weights.iter_mut().enumerate() {
            let reg = if i == BIAS { 0.0 } else { self.l2 * *w };
            *w -= self.lr * (err * x[i] + reg);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn feats(kind: TaskKind) -> Features {
        Features::new(kind)
    }

    #[test]
    fn encode_layout_is_fixed_and_onehot_correct() {
        let x = encode(&feats(TaskKind::CodeEdit), 0);
        assert_eq!(x.len(), FEATURE_DIM);
        assert_eq!(x[BIAS], 1.0);
        assert_eq!(x[task_kind_slot(TaskKind::CodeEdit)], 1.0);
        // exactly one task-kind slot set
        let tk_set: usize = (1..1 + N_TASK_KINDS).filter(|&i| x[i] == 1.0).count();
        assert_eq!(tk_set, 1);
        // exactly one rung slot set
        let rbase = 1 + N_TASK_KINDS + 5;
        let r_set: usize = (rbase..rbase + N_RUNGS).filter(|&i| x[i] == 1.0).count();
        assert_eq!(r_set, 1);
        // rung beyond the width folds into the last slot
        let xr = encode(&feats(TaskKind::Other), 99);
        assert_eq!(xr[rbase + N_RUNGS - 1], 1.0);
    }

    #[test]
    fn predict_stays_in_unit_interval() {
        let mut p = PassPredictor::new(0.5, 0.0);
        for _ in 0..1000 {
            p.update(&feats(TaskKind::Chat), 0, true);
        }
        let v = p.predict(&feats(TaskKind::Chat), 0);
        assert!(v > 0.0 && v < 1.0, "prediction must stay in (0,1): {v}");
    }

    #[test]
    fn converges_to_separate_easy_from_hard_by_rung() {
        // Synthetic separable pattern: rung 0 always FAILS, rung 2 always PASSES for CodeEdit.
        let mut p = PassPredictor::new(0.2, 1e-4);
        let f = feats(TaskKind::CodeEdit);
        for _ in 0..2000 {
            p.update(&f, 0, false);
            p.update(&f, 2, true);
        }
        let low = p.predict(&f, 0);
        let high = p.predict(&f, 2);
        assert!(
            low < 0.2,
            "hopeless rung should predict low pass, got {low}"
        );
        assert!(
            high > 0.8,
            "reliable rung should predict high pass, got {high}"
        );
        assert!(high - low > 0.6, "predictor must separate the two rungs");
    }

    #[test]
    fn l2_keeps_weights_bounded() {
        // Contradictory labels + L2 → weights must not blow up.
        let mut p = PassPredictor::new(0.5, 0.05);
        let f = feats(TaskKind::Review);
        for i in 0..5000 {
            p.update(&f, 1, i % 2 == 0);
        }
        let maxw = p
            .weights
            .iter()
            .cloned()
            .fold(0.0_f64, |a, w| a.max(w.abs()));
        assert!(
            maxw.is_finite() && maxw < 50.0,
            "L2 must bound weights: {maxw}"
        );
        // ambiguous 50/50 label → prediction near 0.5
        let v = p.predict(&f, 1);
        assert!((v - 0.5).abs() < 0.15, "50/50 labels → ~0.5, got {v}");
    }
}
