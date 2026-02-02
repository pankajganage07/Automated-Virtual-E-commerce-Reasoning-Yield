"""
Avery OpsBrain - Streamlit Frontend

A single-page chat interface with HITL approval workflow and memory search.

Run with: streamlit run frontend/app.py
"""

import streamlit as st

from api_client import APIClient, APIError, PendingAction
from components import (
    init_session_state,
    add_message,
    clear_chat,
    render_chat_message,
    render_diagnostics,
    render_hitl_sidebar,
    render_action_result,
)
from config import get_config


# =============================================================================
# Page Configuration
# =============================================================================

config = get_config()

st.set_page_config(
    page_title=config.page_title,
    page_icon=config.page_icon,
    layout="wide",
    initial_sidebar_state="auto",
)

# Initialize session state
init_session_state()


# Initialize API client (cached)
@st.cache_resource
def get_api_client():
    return APIClient(config)


api = get_api_client()


# =============================================================================
# Callbacks
# =============================================================================


def handle_approve(action_id: int):
    """Handle action approval."""
    try:
        with st.spinner(f"Executing action {action_id}..."):
            result = api.approve_action(action_id, execute_immediately=True)

        st.session_state.last_result = result

        # Remove from pending list
        st.session_state.pending_actions = [
            a for a in st.session_state.pending_actions if a.id != action_id
        ]

        # If no more pending actions, resume the query with execution results
        if not st.session_state.pending_actions:
            st.session_state.hitl_waiting = False

            # Resume the graph with execution results for re-synthesis
            if result.success and st.session_state.thread_id:
                with st.spinner("🧠 Analyzing results..."):
                    # Build execution result for the graph
                    execution_results = [
                        {
                            "action_id": result.action_id,
                            "action_type": result.action_type or "unknown",
                            "success": result.success,
                            "message": result.message,
                            "result": result.result,
                        }
                    ]

                    # Resume the graph to re-synthesize with actual data
                    resume_response = api.resume_query(
                        thread_id=st.session_state.thread_id,
                        approved_action_ids=[action_id],
                        execution_results=execution_results,
                    )

                    # Show the re-synthesized answer
                    add_message("assistant", resume_response.answer)
            else:
                # Fallback: show basic success message
                if result.success and result.result:
                    result_data = result.result.get("result", {})
                    if "rows" in result_data:
                        rows = result_data["rows"]
                        add_message(
                            "assistant",
                            f"✅ **Action Executed Successfully**\n\n"
                            f"Found {len(rows)} result(s). See the data below.",
                        )
                    else:
                        add_message("assistant", f"✅ **Action Executed:** {result.message}")
                else:
                    add_message("assistant", f"❌ **Action Failed:** {result.message}")
        else:
            # More pending actions, just show execution status
            if result.success and result.result:
                result_data = result.result.get("result", {})
                if "rows" in result_data:
                    rows = result_data["rows"]
                    add_message(
                        "assistant",
                        f"✅ **Action Executed Successfully**\n\n"
                        f"Found {len(rows)} result(s). More actions pending approval.",
                    )
                else:
                    add_message("assistant", f"✅ **Action Executed:** {result.message}")
            else:
                add_message("assistant", f"❌ **Action Failed:** {result.message}")

        st.rerun()

    except APIError as e:
        st.error(f"Failed to approve action: {e.message}")


def handle_reject(action_id: int):
    """Handle action rejection."""
    try:
        result = api.reject_action(action_id)

        # Remove from pending list
        st.session_state.pending_actions = [
            a for a in st.session_state.pending_actions if a.id != action_id
        ]

        # If no more pending actions, clear HITL waiting state
        if not st.session_state.pending_actions:
            st.session_state.hitl_waiting = False

        add_message("assistant", f"❌ **Action Rejected:** {result.message}")
        st.rerun()

    except APIError as e:
        st.error(f"Failed to reject action: {e.message}")


def handle_query(question: str):
    """Handle a new user query."""
    # Add user message to chat
    add_message("user", question)

    try:
        with st.spinner("🧠 Thinking..."):
            # Pass conversation history for context resolution
            # Get last 10 messages (5 turns) for context
            history = st.session_state.messages[-10:] if st.session_state.messages else None

            response = api.query(
                question=question,
                thread_id=st.session_state.thread_id,
                history=history,
            )

        # Update thread ID
        st.session_state.thread_id = response.thread_id

        # Add assistant response to chat
        add_message("assistant", response.answer)

        # Handle HITL
        if response.hitl_waiting and response.pending_actions:
            st.session_state.hitl_waiting = True
            st.session_state.pending_actions = response.pending_actions
            add_message(
                "assistant",
                "⚠️ **Action Required:** I have recommendations that need your approval. "
                "Please review them in the sidebar.",
            )
        else:
            st.session_state.hitl_waiting = False
            st.session_state.pending_actions = []

        st.rerun()

    except APIError as e:
        st.error(f"Query failed: {e.message}")


# =============================================================================
# Main Layout
# =============================================================================

# Header
st.title(f"{config.page_icon} {config.page_title}")
st.caption("AI-powered E-commerce Operations Assistant")

# Sidebar - HITL Approvals
with st.sidebar:
    # HITL Section
    if st.session_state.hitl_waiting and st.session_state.pending_actions:
        render_hitl_sidebar(
            st.session_state.pending_actions,
            on_approve=handle_approve,
            on_reject=handle_reject,
        )
        st.divider()

    # Clear Chat Button
    if st.button("🗑️ Clear Chat", use_container_width=True):
        clear_chat()
        st.rerun()


# Main Chat Area
chat_container = st.container()

with chat_container:
    # Display chat history
    for msg in st.session_state.messages:
        render_chat_message(msg["role"], msg["content"])

    # Show last action result if present
    if st.session_state.last_result:
        with st.container():
            render_action_result(st.session_state.last_result)
            st.session_state.last_result = None  # Clear after displaying

# Chat Input
if prompt := st.chat_input(
    "Ask about sales, inventory, marketing, or support...",
    disabled=st.session_state.hitl_waiting,
):
    handle_query(prompt)

# HITL Warning Banner
if st.session_state.hitl_waiting:
    st.warning(
        "⏸️ **Waiting for Approval** - Please review and approve/reject the pending "
        "actions in the sidebar before continuing."
    )


# =============================================================================
# Footer
# =============================================================================

st.divider()
st.caption(
    "Avery OpsBrain | Powered by LangGraph & GPT-4 | "
    f"Thread: {st.session_state.thread_id or 'New Session'}"
)
