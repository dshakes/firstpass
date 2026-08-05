//! Last-resort context condensing: drop the middle of a conversation that will not fit anywhere.
//!
//! ## Why this is narrow on purpose
//!
//! Condensing routinely — trimming every long conversation to save tokens — would mean the gate
//! verifies an answer produced from a prompt the **client never sent**. The receipt would attest to
//! a decision about a different question than the one asked, which is the one thing this product
//! cannot afford to get wrong.
//!
//! So it fires in exactly one situation: the prompt has already overflowed the context window of
//! **every rung on the ladder**, and the alternative is not a cheaper answer, it is *no answer*.
//! At that point the trade is no longer "faithful vs. degraded" but "degraded vs. failed", and a
//! degraded answer the receipt openly labels is the better of the two.
//!
//! ## What it keeps
//!
//! The head and the tail. The opening turns usually carry the task definition and the tail carries
//! the live thread; the middle is where a long agent conversation accumulates tool output nobody
//! refers to again. A marker turn is inserted where the elision happened, so the model is told the
//! history is incomplete rather than being left to infer it from a discontinuity.

use crate::provider::{ChatMessage, ModelRequest};

/// Outcome of a condensing attempt.
#[derive(Debug, Clone)]
pub struct Condensed {
    /// The request with its middle removed.
    pub request: ModelRequest,
    /// How many turns were dropped — recorded on the receipt, never inferred.
    pub dropped: usize,
}

/// Drop the middle of `req`'s conversation, keeping `keep_head` opening turns and `keep_tail`
/// closing ones.
///
/// Returns `None` when there is nothing useful to drop, so a caller cannot loop forever retrying a
/// request that is not getting any smaller — a prompt that overflows on a single enormous turn
/// cannot be fixed by removing turns, and pretending otherwise would just spend another call.
#[must_use]
pub fn condense(req: &ModelRequest, keep_head: usize, keep_tail: usize) -> Option<Condensed> {
    let n = req.messages.len();
    // Need at least one turn strictly between the head and tail slices to be worth a retry.
    let kept = keep_head.saturating_add(keep_tail);
    if n <= kept.saturating_add(1) {
        return None;
    }
    let dropped = n - kept;

    let mut messages: Vec<ChatMessage> = Vec::with_capacity(kept + 1);
    messages.extend(req.messages.iter().take(keep_head).cloned());
    // Tell the model its history is incomplete. Without this it sees a discontinuity and has to
    // guess whether it misremembers — which is how condensing turns into confabulation.
    messages.push(ChatMessage::text(
        "user",
        format!(
            "[{dropped} earlier turns omitted: this conversation exceeded the context window of \
             every available model. The opening and the most recent turns are shown. If answering \
             depends on the omitted portion, say so rather than guessing.]"
        ),
    ));
    messages.extend(req.messages.iter().skip(n - keep_tail).cloned());

    let mut out = req.clone();
    out.messages = messages;
    // The raw body no longer describes what is being sent, and a provider that forwards `raw`
    // verbatim would re-send the full conversation and overflow again. Clearing it forces the
    // normalized-field path, which is built from the condensed `messages`.
    out.raw = serde_json::Value::Null;
    Some(Condensed {
        request: out,
        dropped,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(n: usize) -> ModelRequest {
        ModelRequest {
            model: "anthropic/claude-haiku-4-5".to_owned(),
            system: Some("sys".to_owned()),
            messages: (0..n)
                .map(|i| ChatMessage::text("user", format!("turn {i}")))
                .collect(),
            max_tokens: 1024,
            tools: serde_json::Value::Null,
            raw: serde_json::json!({ "messages": "the full original" }),
        }
    }

    #[test]
    fn the_head_and_tail_survive_and_the_middle_goes() {
        let c = condense(&req(20), 2, 3).expect("20 turns can be condensed");

        assert_eq!(c.dropped, 15);
        // 2 head + 1 marker + 3 tail
        assert_eq!(c.request.messages.len(), 6);
        assert_eq!(c.request.messages[0].text_view(), "turn 0");
        assert_eq!(c.request.messages[1].text_view(), "turn 1");
        assert!(
            c.request.messages[2]
                .text_view()
                .contains("15 earlier turns omitted")
        );
        assert_eq!(c.request.messages[5].text_view(), "turn 19");
    }

    #[test]
    fn the_model_is_told_the_history_is_incomplete() {
        // Without the marker the model sees a discontinuity and has to guess whether it
        // misremembers, which is how condensing turns into confabulation.
        let c = condense(&req(30), 1, 1).expect("condensable");
        let marker = c.request.messages[1].text_view();
        assert!(marker.contains("omitted"), "{marker}");
        assert!(
            marker.contains("say so rather than guessing"),
            "the model needs an instruction, not just a notice: {marker}"
        );
    }

    #[test]
    fn the_raw_body_is_cleared_or_the_retry_overflows_again() {
        // Anthropic-dialect providers forward `raw` verbatim. Left in place, the retry would send
        // the FULL conversation again and overflow identically — an extra call for nothing.
        let c = condense(&req(20), 2, 2).expect("condensable");
        assert!(
            c.request.raw.is_null(),
            "raw must not survive condensing: {:?}",
            c.request.raw
        );
    }

    #[test]
    fn a_conversation_with_no_droppable_middle_returns_none() {
        // The caller retries on Some. A request that is not getting smaller must return None, or
        // the retry loop spends another call on the same overflow.
        assert!(
            condense(&req(5), 2, 3).is_none(),
            "nothing between head and tail"
        );
        assert!(
            condense(&req(6), 2, 3).is_none(),
            "one turn is not worth a retry"
        );
        assert!(condense(&req(1), 2, 3).is_none());
        assert!(condense(&req(7), 2, 3).is_some(), "two spare turns is");
    }

    #[test]
    fn the_system_prompt_and_limits_are_untouched() {
        // Condensing drops history, not the task definition or the caller's ceilings.
        let c = condense(&req(20), 2, 2).expect("condensable");
        assert_eq!(c.request.system.as_deref(), Some("sys"));
        assert_eq!(c.request.max_tokens, 1024);
        assert_eq!(c.request.model, "anthropic/claude-haiku-4-5");
    }
}
