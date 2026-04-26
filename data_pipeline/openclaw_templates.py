"""
OpenCLaw Template Manager
=========================
Downloads and manages open-source legal contract templates from OpenCLaw
and other open-source legal template repositories.

These templates are used by the Drafting Agent to generate contracts.
"""

import json
import os
from pathlib import Path

import requests
from rich.console import Console

console = Console()

# Built-in contract templates (these are common open-source legal templates)
# In a full version, these would be fetched from OpenCLaw's API/GitHub
BUILT_IN_TEMPLATES = {
    "nda_mutual": {
        "name": "Mutual Non-Disclosure Agreement",
        "category": "Confidentiality",
        "jurisdiction": "General / US",
        "template": """
# MUTUAL NON-DISCLOSURE AGREEMENT

**Effective Date:** {{effective_date}}

This Mutual Non-Disclosure Agreement ("Agreement") is entered into by and between:

**Party A:** {{party_a_name}}, a {{party_a_type}} organized under the laws of {{party_a_jurisdiction}}, with its principal place of business at {{party_a_address}} ("Disclosing Party")

**Party B:** {{party_b_name}}, a {{party_b_type}} organized under the laws of {{party_b_jurisdiction}}, with its principal place of business at {{party_b_address}} ("Receiving Party")

(collectively, the "Parties")

## 1. PURPOSE
The Parties wish to explore a potential business relationship concerning {{purpose}} (the "Purpose"). In connection with the Purpose, each Party may disclose to the other certain confidential and proprietary information.

## 2. DEFINITION OF CONFIDENTIAL INFORMATION
"Confidential Information" means any information disclosed by either Party to the other Party, either directly or indirectly, in writing, orally, or by inspection of tangible objects, that is designated as "Confidential," "Proprietary," or some similar designation, or that reasonably should be understood to be confidential given the nature of the information and circumstances of disclosure.

Confidential Information includes, but is not limited to:
- Trade secrets, inventions, ideas, processes, formulas, source and object code, data, programs, other works of authorship, know-how, improvements, discoveries, developments, designs, and techniques
- Business plans, financial information, customer lists, and marketing strategies
- Technical specifications, documentation, and research data

## 3. EXCLUSIONS
Confidential Information shall not include information that:
(a) is or becomes publicly available without breach of this Agreement;
(b) was known to the Receiving Party prior to disclosure;
(c) is independently developed by the Receiving Party without use of Confidential Information;
(d) is rightfully received from a third party without restriction on disclosure.

## 4. OBLIGATIONS
Each Party agrees to:
(a) Hold the Confidential Information in strict confidence;
(b) Not disclose the Confidential Information to any third parties without prior written consent;
(c) Use the Confidential Information solely for the Purpose;
(d) Protect the Confidential Information using at least the same degree of care used to protect its own confidential information, but no less than reasonable care.

## 5. TERM
This Agreement shall remain in effect for {{term_years}} year(s) from the Effective Date, unless earlier terminated by either Party with {{notice_days}} days' written notice.

## 6. RETURN OF MATERIALS
Upon termination of this Agreement or upon request, each Party shall promptly return or destroy all Confidential Information and any copies thereof.

## 7. GOVERNING LAW
This Agreement shall be governed by and construed in accordance with the laws of {{governing_law_jurisdiction}}, without regard to its conflict of laws provisions.

## 8. REMEDIES
Each Party acknowledges that any breach of this Agreement may cause irreparable harm and that monetary damages may be insufficient. The non-breaching Party shall be entitled to seek equitable relief, including injunction and specific performance, in addition to all other remedies available at law or in equity.

## 9. ENTIRE AGREEMENT
This Agreement constitutes the entire agreement between the Parties concerning the subject matter hereof and supersedes all prior agreements and understandings.

---

**PARTY A:**
Signature: ________________________
Name: {{party_a_signatory}}
Title: {{party_a_title}}
Date: ________________________

**PARTY B:**
Signature: ________________________
Name: {{party_b_signatory}}
Title: {{party_b_title}}
Date: ________________________
""",
        "variables": [
            "effective_date", "party_a_name", "party_a_type", "party_a_jurisdiction",
            "party_a_address", "party_b_name", "party_b_type", "party_b_jurisdiction",
            "party_b_address", "purpose", "term_years", "notice_days",
            "governing_law_jurisdiction", "party_a_signatory", "party_a_title",
            "party_b_signatory", "party_b_title",
        ],
    },
    "saas_agreement": {
        "name": "SaaS Service Agreement",
        "category": "Technology",
        "jurisdiction": "General / US / EU",
        "template": """
# SOFTWARE AS A SERVICE (SaaS) AGREEMENT

**Effective Date:** {{effective_date}}

This SaaS Agreement ("Agreement") is entered into between:

**Provider:** {{provider_name}}, {{provider_address}} ("Provider")
**Customer:** {{customer_name}}, {{customer_address}} ("Customer")

## 1. SERVICES
Provider agrees to provide Customer with access to {{service_description}} (the "Service") as described in Exhibit A (Service Level Agreement).

## 2. SUBSCRIPTION TERM
The initial subscription term shall be {{initial_term}} commencing on the Effective Date ("Initial Term"). Thereafter, the subscription shall automatically renew for successive {{renewal_term}} periods unless either Party provides written notice of non-renewal at least {{notice_days}} days prior to the end of the then-current term.

## 3. FEES AND PAYMENT
(a) Customer shall pay Provider {{fee_amount}} per {{fee_period}} ("Subscription Fee").
(b) All fees are due within {{payment_terms}} days of invoice date.
(c) Late payments shall accrue interest at the rate of {{late_interest_rate}}% per month.

## 4. DATA PROTECTION & GDPR COMPLIANCE
(a) Provider shall process Customer's personal data only in accordance with Customer's documented instructions.
(b) Provider shall implement appropriate technical and organizational measures to ensure a level of security appropriate to the risk.
(c) Provider shall notify Customer without undue delay upon becoming aware of a personal data breach.
(d) A Data Processing Agreement (DPA) in compliance with GDPR Article 28 is attached as Exhibit B.

## 5. SERVICE LEVEL AGREEMENT
(a) Provider guarantees {{uptime_percentage}}% monthly uptime.
(b) Scheduled maintenance windows: {{maintenance_window}}.
(c) Service credits for downtime exceeding SLA: {{credit_terms}}.

## 6. INTELLECTUAL PROPERTY
(a) Provider retains all rights to the Service and underlying technology.
(b) Customer retains all rights to Customer Data.
(c) Provider is granted a limited license to use Customer Data solely to provide the Service.

## 7. LIMITATION OF LIABILITY
(a) NEITHER PARTY SHALL BE LIABLE FOR ANY INDIRECT, INCIDENTAL, SPECIAL, CONSEQUENTIAL, OR PUNITIVE DAMAGES.
(b) Provider's total aggregate liability shall not exceed {{liability_cap}}.

## 8. TERMINATION
Either Party may terminate this Agreement:
(a) For material breach, with {{cure_period}} days' written notice and opportunity to cure;
(b) Immediately if the other Party becomes insolvent or files for bankruptcy.

## 9. GOVERNING LAW
This Agreement shall be governed by the laws of {{governing_law}}.

## 10. DISPUTE RESOLUTION
Any disputes shall be resolved through {{dispute_resolution}} in {{dispute_venue}}.

---

**PROVIDER:** {{provider_name}}
By: ________________________  Date: ____________

**CUSTOMER:** {{customer_name}}
By: ________________________  Date: ____________
""",
        "variables": [
            "effective_date", "provider_name", "provider_address",
            "customer_name", "customer_address", "service_description",
            "initial_term", "renewal_term", "notice_days", "fee_amount",
            "fee_period", "payment_terms", "late_interest_rate",
            "uptime_percentage", "maintenance_window", "credit_terms",
            "liability_cap", "cure_period", "governing_law",
            "dispute_resolution", "dispute_venue",
        ],
    },
    "employment_contract_de": {
        "name": "Employment Contract (German Law / Arbeitsvertrag)",
        "category": "Employment",
        "jurisdiction": "Germany",
        "template": """
# ARBEITSVERTRAG / EMPLOYMENT CONTRACT

**Datum / Date:** {{effective_date}}

Zwischen / Between:

**Arbeitgeber / Employer:** {{employer_name}}, {{employer_address}} ("Arbeitgeber")
**Arbeitnehmer / Employee:** {{employee_name}}, {{employee_address}} ("Arbeitnehmer")

## § 1 BEGINN UND DAUER / COMMENCEMENT AND DURATION
Das Arbeitsverhältnis beginnt am {{start_date}}.
{{#if probation_period}}
Die Probezeit beträgt {{probation_period}} Monate. Während der Probezeit kann das Arbeitsverhältnis mit einer Frist von zwei Wochen gekündigt werden.
{{/if}}
Das Arbeitsverhältnis wird auf {{contract_duration}} geschlossen.

## § 2 TÄTIGKEIT / DUTIES
Der Arbeitnehmer wird als {{job_title}} eingestellt. Der Arbeitsort ist {{work_location}}.

## § 3 ARBEITSZEIT / WORKING HOURS
Die regelmäßige wöchentliche Arbeitszeit beträgt {{weekly_hours}} Stunden.

## § 4 VERGÜTUNG / COMPENSATION
(a) Das monatliche Bruttogehalt beträgt EUR {{monthly_salary}}.
(b) Die Zahlung erfolgt jeweils zum {{payment_day}}. des Monats.
{{#if bonus}}
(c) Zusätzlich erhält der Arbeitnehmer {{bonus_terms}}.
{{/if}}

## § 5 URLAUB / VACATION
Der Arbeitnehmer hat Anspruch auf {{vacation_days}} Arbeitstage Urlaub pro Kalenderjahr.

## § 6 KÜNDIGUNG / TERMINATION
Nach Ablauf der Probezeit gelten die gesetzlichen Kündigungsfristen gemäß § 622 BGB. Die Kündigung bedarf der Schriftform.

## § 7 GEHEIMHALTUNG / CONFIDENTIALITY
Der Arbeitnehmer verpflichtet sich, über alle Geschäfts- und Betriebsgeheimnisse Stillschweigen zu bewahren, auch nach Beendigung des Arbeitsverhältnisses.

## § 8 NEBENTÄTIGKEIT / SECONDARY EMPLOYMENT
Jede entgeltliche Nebentätigkeit bedarf der vorherigen schriftlichen Zustimmung des Arbeitgebers.

## § 9 ANWENDBARES RECHT / GOVERNING LAW
Es gilt deutsches Recht. Gerichtsstand ist {{court_jurisdiction}}.

---

**Arbeitgeber / Employer:**
Unterschrift: ________________________  Datum: ____________

**Arbeitnehmer / Employee:**
Unterschrift: ________________________  Datum: ____________
""",
        "variables": [
            "effective_date", "employer_name", "employer_address",
            "employee_name", "employee_address", "start_date",
            "probation_period", "contract_duration", "job_title",
            "work_location", "weekly_hours", "monthly_salary",
            "payment_day", "bonus_terms", "vacation_days", "court_jurisdiction",
        ],
    },
    "consulting_agreement": {
        "name": "Consulting Services Agreement",
        "category": "Services",
        "jurisdiction": "General",
        "template": """
# CONSULTING SERVICES AGREEMENT

**Effective Date:** {{effective_date}}

**Client:** {{client_name}} ("Client")
**Consultant:** {{consultant_name}} ("Consultant")

## 1. SERVICES
Consultant agrees to provide {{service_description}} (the "Services") as described in the Statement of Work attached as Exhibit A.

## 2. TERM
This Agreement commences on {{start_date}} and continues until {{end_date}}, unless earlier terminated.

## 3. COMPENSATION
(a) Client shall pay Consultant {{rate_amount}} per {{rate_unit}}.
(b) Consultant shall submit invoices {{invoice_frequency}}.
(c) Payment due within {{payment_terms}} days of invoice.

## 4. INDEPENDENT CONTRACTOR
Consultant is an independent contractor and not an employee of Client.

## 5. INTELLECTUAL PROPERTY
{{ip_ownership_clause}}

## 6. CONFIDENTIALITY
Consultant shall maintain the confidentiality of all Client information.

## 7. TERMINATION
Either Party may terminate with {{notice_period}} days' written notice.

## 8. GOVERNING LAW
Governed by the laws of {{governing_law}}.

---

**CLIENT:** ________________________  Date: ____________
**CONSULTANT:** ________________________  Date: ____________
""",
        "variables": [
            "effective_date", "client_name", "consultant_name",
            "service_description", "start_date", "end_date",
            "rate_amount", "rate_unit", "invoice_frequency",
            "payment_terms", "ip_ownership_clause", "notice_period",
            "governing_law",
        ],
    },
    "dpa_gdpr": {
        "name": "Data Processing Agreement (GDPR Art. 28)",
        "category": "Privacy / Data Protection",
        "jurisdiction": "EU / Germany",
        "template": """
# DATA PROCESSING AGREEMENT
## pursuant to Art. 28 GDPR

**Between:**
**Controller:** {{controller_name}}, {{controller_address}} ("Controller")
**Processor:** {{processor_name}}, {{processor_address}} ("Processor")

## 1. SUBJECT MATTER AND DURATION
This DPA governs the processing of personal data by Processor on behalf of Controller in connection with {{main_agreement_reference}}.

## 2. NATURE AND PURPOSE OF PROCESSING
{{processing_purpose}}

## 3. TYPE OF PERSONAL DATA
{{data_types}}

## 4. CATEGORIES OF DATA SUBJECTS
{{data_subject_categories}}

## 5. OBLIGATIONS OF THE PROCESSOR
The Processor shall:
(a) Process personal data only on documented instructions from the Controller;
(b) Ensure persons authorized to process have committed to confidentiality;
(c) Implement appropriate technical and organizational measures (Art. 32 GDPR);
(d) Not engage sub-processors without prior written authorization;
(e) Assist the Controller in responding to data subject requests;
(f) Delete or return all personal data upon termination;
(g) Make available all information necessary to demonstrate compliance.

## 6. SUB-PROCESSORS
{{sub_processor_terms}}

## 7. INTERNATIONAL TRANSFERS
Personal data shall not be transferred outside the EEA without appropriate safeguards as per Chapter V GDPR.

## 8. DATA BREACH NOTIFICATION
Processor shall notify Controller without undue delay (and in any event within {{breach_notification_hours}} hours) after becoming aware of a personal data breach.

## 9. TECHNICAL AND ORGANIZATIONAL MEASURES
See Annex I for detailed security measures.

---

**CONTROLLER:** ________________________  Date: ____________
**PROCESSOR:** ________________________  Date: ____________
""",
        "variables": [
            "controller_name", "controller_address", "processor_name",
            "processor_address", "main_agreement_reference", "processing_purpose",
            "data_types", "data_subject_categories", "sub_processor_terms",
            "breach_notification_hours",
        ],
    },
}


def get_template_catalog():
    """Return the catalog of available templates."""
    catalog = []
    for key, template in BUILT_IN_TEMPLATES.items():
        catalog.append({
            "id": key,
            "name": template["name"],
            "category": template["category"],
            "jurisdiction": template["jurisdiction"],
            "variables": template["variables"],
        })
    return catalog


def get_template(template_id: str) -> dict | None:
    """Get a specific template by ID."""
    return BUILT_IN_TEMPLATES.get(template_id)


def fill_template(template_id: str, variables: dict) -> str:
    """Fill a template with provided variables."""
    template_data = BUILT_IN_TEMPLATES.get(template_id)
    if not template_data:
        return f"Template '{template_id}' not found."

    content = template_data["template"]

    for var_name, var_value in variables.items():
        placeholder = "{{" + var_name + "}}"
        content = content.replace(placeholder, str(var_value))

    return content


def save_templates(output_dir="data/processed"):
    """Save template catalog to disk."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    catalog = get_template_catalog()
    with open(output_path / "template_catalog.json", "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2)

    # Save full templates
    with open(output_path / "templates_full.json", "w", encoding="utf-8") as f:
        json.dump(BUILT_IN_TEMPLATES, f, indent=2, ensure_ascii=False)

    console.print(f"[green]✓ Saved {len(catalog)} contract templates[/]")
    return catalog


if __name__ == "__main__":
    catalog = save_templates()
    for t in catalog:
        console.print(f"  📄 {t['name']} ({t['jurisdiction']})")
