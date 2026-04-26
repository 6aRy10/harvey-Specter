"""
Partner Approval Workflow via Slack
====================================
Sends AI-drafted outputs to a Slack channel for partner review.
The partner can approve or reject before anything is sent to the client.

Requires: SLACK_WEBHOOK_URL in .env
Optional: SLACK_APPROVAL_CHANNEL (defaults to #partner-approvals)
"""

import os
import json
import uuid
import time
from datetime import datetime
from typing import Optional

import requests


# In-memory approval store (production: use Redis/DB)
_pending_approvals: dict = {}


class ApprovalRequest:
    """Represents a pending partner approval."""

    def __init__(self, matter_id: str, doc_type: str, content: str, recipient: str = "client"):
        self.id = str(uuid.uuid4())[:8]
        self.matter_id = matter_id
        self.doc_type = doc_type  # "contract_draft", "legal_memo", "review_report"
        self.content = content
        self.recipient = recipient
        self.status = "pending"  # pending | approved | rejected
        self.created_at = datetime.utcnow().isoformat()
        self.reviewed_at = None
        self.reviewer_note = None


def get_webhook_url() -> Optional[str]:
    return os.getenv("SLACK_WEBHOOK_URL", "")


def is_configured() -> bool:
    return bool(get_webhook_url())


def send_approval_request(
    matter_id: str,
    doc_type: str,
    content: str,
    summary: str = "",
    recipient: str = "client",
) -> dict:
    """
    Send a Slack message asking the partner to approve/reject a draft.
    Returns the approval request object.
    """
    approval = ApprovalRequest(matter_id, doc_type, content, recipient)
    _pending_approvals[approval.id] = approval

    webhook_url = get_webhook_url()

    # Truncate content for Slack preview
    preview = content[:500] + ("..." if len(content) > 500 else "")

    # Format the Slack message
    slack_message = {
        "blocks": [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"🔒 Partner Approval Required — {doc_type.replace('_', ' ').title()}",
                    "emoji": True,
                },
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Matter ID:*\n`{matter_id}`"},
                    {"type": "mrkdwn", "text": f"*Approval ID:*\n`{approval.id}`"},
                    {"type": "mrkdwn", "text": f"*Document Type:*\n{doc_type.replace('_', ' ').title()}"},
                    {"type": "mrkdwn", "text": f"*Recipient:*\n{recipient}"},
                ],
            },
            {"type": "divider"},
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Summary:*\n{summary or 'AI-generated document ready for review.'}",
                },
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Preview:*\n```{preview}```",
                },
            },
            {"type": "divider"},
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        "✅ *To approve:* call `POST /api/approvals/{id}/approve`\n"
                        "❌ *To reject:* call `POST /api/approvals/{id}/reject`\n\n"
                        f"_Or use the dashboard to approve/reject approval `{approval.id}`_"
                    ).format(id=approval.id),
                },
            },
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f"⏰ {approval.created_at} UTC | Harveyy AI Law Firm",
                    }
                ],
            },
        ]
    }

    # Send to Slack
    sent = False
    slack_error = None
    if webhook_url:
        try:
            resp = requests.post(
                webhook_url,
                json=slack_message,
                headers={"Content-Type": "application/json"},
                timeout=10,
            )
            sent = resp.status_code == 200
            if not sent:
                slack_error = f"Slack returned {resp.status_code}: {resp.text}"
        except Exception as e:
            slack_error = str(e)

    return {
        "approval_id": approval.id,
        "matter_id": matter_id,
        "doc_type": doc_type,
        "status": "pending",
        "slack_sent": sent,
        "slack_error": slack_error,
        "created_at": approval.created_at,
        "message": f"Approval request sent. Waiting for partner review. (ID: {approval.id})",
    }


def approve(approval_id: str, note: str = "") -> dict:
    """Partner approves a pending request."""
    if approval_id not in _pending_approvals:
        return {"error": f"Approval {approval_id} not found"}
    a = _pending_approvals[approval_id]
    if a.status != "pending":
        return {"error": f"Approval already {a.status}"}
    a.status = "approved"
    a.reviewed_at = datetime.utcnow().isoformat()
    a.reviewer_note = note
    return {
        "approval_id": a.id,
        "status": "approved",
        "reviewed_at": a.reviewed_at,
        "note": note,
        "content": a.content,
    }


def reject(approval_id: str, note: str = "") -> dict:
    """Partner rejects a pending request."""
    if approval_id not in _pending_approvals:
        return {"error": f"Approval {approval_id} not found"}
    a = _pending_approvals[approval_id]
    if a.status != "pending":
        return {"error": f"Approval already {a.status}"}
    a.status = "rejected"
    a.reviewed_at = datetime.utcnow().isoformat()
    a.reviewer_note = note
    return {
        "approval_id": a.id,
        "status": "rejected",
        "reviewed_at": a.reviewed_at,
        "note": note,
    }


def list_pending() -> list:
    """List all pending approvals."""
    return [
        {
            "approval_id": a.id,
            "matter_id": a.matter_id,
            "doc_type": a.doc_type,
            "recipient": a.recipient,
            "status": a.status,
            "created_at": a.created_at,
            "content_preview": a.content[:200],
        }
        for a in _pending_approvals.values()
    ]


def get_approval(approval_id: str) -> Optional[dict]:
    """Get a specific approval by ID."""
    if approval_id not in _pending_approvals:
        return None
    a = _pending_approvals[approval_id]
    return {
        "approval_id": a.id,
        "matter_id": a.matter_id,
        "doc_type": a.doc_type,
        "recipient": a.recipient,
        "status": a.status,
        "created_at": a.created_at,
        "reviewed_at": a.reviewed_at,
        "reviewer_note": a.reviewer_note,
        "content": a.content,
    }
