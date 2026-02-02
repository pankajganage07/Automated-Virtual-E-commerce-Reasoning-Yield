"""
Reusable Streamlit UI components for Avery OpsBrain.
"""

from __future__ import annotations

import json
import streamlit as st
from typing import Callable

from api_client import PendingAction, ActionResult, HistoryItem


def render_chat_message(role: str, content: str, avatar: str | None = None):
    """
    Render a chat message.

    Args:
        role: 'user' or 'assistant'
        content: Message content (supports markdown)
        avatar: Optional avatar emoji/image
    """
    if avatar is None:
        avatar = "👤" if role == "user" else "🧠"

    with st.chat_message(role, avatar=avatar):
        st.markdown(content)


def render_diagnostics(diagnostics: list[str]):
    """Render diagnostics in a collapsible section."""
    if not diagnostics:
        return

    with st.expander("🔍 Diagnostics", expanded=False):
        for diag in diagnostics:
            st.text(f"• {diag}")


def render_pending_action_card(
    action: PendingAction,
    on_approve: Callable[[int], None],
    on_reject: Callable[[int], None],
    key_prefix: str = "",
):
    """
    Render a pending action card with approve/reject buttons.

    Args:
        action: The pending action to display
        on_approve: Callback when approved (receives action_id)
        on_reject: Callback when rejected (receives action_id)
        key_prefix: Prefix for unique widget keys
    """
    with st.container():
        # Header with action type and agent
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**{action.action_type.replace('_', ' ').title()}**")
            st.caption(f"Agent: {action.agent} | ID: {action.id}")
        with col2:
            status_colors = {
                "pending": "🟡",
                "approved": "🟢",
                "rejected": "🔴",
                "executed": "✅",
            }
            st.markdown(f"{status_colors.get(action.status, '⚪')} {action.status.title()}")

        # Reasoning
        st.markdown(f"📝 *{action.reasoning[:200]}{'...' if len(action.reasoning) > 200 else ''}*")

        # Payload in expander
        with st.expander("View Payload", expanded=False):
            st.json(action.payload)

        # Action buttons (only show for pending actions)
        if action.status == "pending":
            col_approve, col_reject, col_spacer = st.columns([1, 1, 2])
            with col_approve:
                if st.button("✅ Approve", key=f"{key_prefix}_approve_{action.id}", type="primary"):
                    on_approve(action.id)
            with col_reject:
                if st.button("❌ Reject", key=f"{key_prefix}_reject_{action.id}"):
                    on_reject(action.id)

        st.divider()


def render_action_result(result: ActionResult):
    """Render the result of an action execution."""
    if result.success:
        st.success(f"✅ {result.message}")
        if result.result:
            with st.expander("View Result", expanded=True):
                # Format SQL results nicely if present
                if "rows" in result.result.get("result", {}):
                    rows = result.result["result"]["rows"]
                    if rows:
                        st.dataframe(rows, use_container_width=True)
                    else:
                        st.info("No rows returned")
                else:
                    st.json(result.result)
    else:
        st.error(f"❌ {result.message}")


def render_history_item(item: HistoryItem, show_score: bool = False):
    """
    Render a history/memory item.

    Args:
        item: The history item to display
        show_score: Whether to show similarity score (for search results)
    """
    with st.container():
        # Header
        header = (
            f"📋 {item.incident_summary[:80]}{'...' if len(item.incident_summary) > 80 else ''}"
        )
        if show_score and item.score is not None:
            header += f" (Score: {item.score:.2f})"
        st.markdown(f"**{header}**")

        if item.created_at:
            st.caption(f"Created: {item.created_at}")

        # Details
        cols = st.columns(2)
        with cols[0]:
            if item.root_cause:
                st.markdown(f"**Root Cause:** {item.root_cause[:100]}...")
        with cols[1]:
            if item.action_taken:
                st.markdown(f"**Action Taken:** {item.action_taken[:100]}...")

        if item.outcome:
            st.markdown(f"**Outcome:** {item.outcome}")

        st.divider()


def render_hitl_sidebar(
    pending_actions: list[PendingAction],
    on_approve: Callable[[int], None],
    on_reject: Callable[[int], None],
):
    """
    Render the HITL approval sidebar.

    Args:
        pending_actions: List of pending actions to display
        on_approve: Callback when an action is approved
        on_reject: Callback when an action is rejected
    """
    with st.sidebar:
        st.header("🔐 Pending Approvals")

        if not pending_actions:
            st.info("No actions pending approval")
            return

        st.warning(f"⚠️ {len(pending_actions)} action(s) awaiting your approval")

        for i, action in enumerate(pending_actions):
            render_pending_action_card(
                action,
                on_approve=on_approve,
                on_reject=on_reject,
                key_prefix=f"sidebar_{i}",
            )


def render_memory_search(search_fn: Callable[[str], list[HistoryItem]]):
    """
    Render the memory search component.

    Args:
        search_fn: Function to call for searching (receives query string)
    """
    st.subheader("🔎 Search Past Incidents")

    query = st.text_input(
        "Search query",
        placeholder="e.g., low stock electronics",
        key="memory_search_input",
    )

    if query and len(query) >= 3:
        with st.spinner("Searching..."):
            results = search_fn(query)

        if results:
            st.success(f"Found {len(results)} similar incident(s)")
            for item in results:
                render_history_item(item, show_score=True)
        else:
            st.info("No matching incidents found")
    elif query:
        st.caption("Enter at least 3 characters to search")


def init_session_state():
    """Initialize Streamlit session state with defaults."""
    defaults = {
        "messages": [],  # Chat history
        "thread_id": None,  # Current conversation thread
        "pending_actions": [],  # Actions awaiting approval
        "hitl_waiting": False,  # Whether we're waiting for HITL
        "last_result": None,  # Last action execution result
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def add_message(role: str, content: str):
    """Add a message to the chat history."""
    st.session_state.messages.append(
        {
            "role": role,
            "content": content,
        }
    )


def clear_chat():
    """Clear the chat history and reset state."""
    st.session_state.messages = []
    st.session_state.thread_id = None
    st.session_state.pending_actions = []
    st.session_state.hitl_waiting = False
    st.session_state.last_result = None
