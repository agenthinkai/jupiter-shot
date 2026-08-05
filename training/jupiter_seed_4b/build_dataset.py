#!/usr/bin/env python3
"""
Jupiter Seed 4B — Gold Dataset Builder
========================================
Generates 850 provenance-controlled, internally authored examples across
three mutually exclusive splits:
  - train:  600 examples
  - valid:  100 examples
  - eval:   150 examples

All examples are internally authored. No teacher-model outputs, no paid APIs,
no copyrighted material, no confidential client data.

Usage:
    python3 build_dataset.py --output-dir training/jupiter_seed_4b/data [--seed 42]
    python3 build_dataset.py --help
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

SCHEMA_VERSION = "1.0"
AUTHOR_TYPE = "internal_human_author"
SOURCE_TYPE = "internally_authored"
SOURCE_NAME = "AgenThink Jupiter Seed 4B Dataset"
SOURCE_URL = "https://github.com/agenthinkai/jupiter-shot"
LICENCE = "AgenThink Internal — Not for redistribution"
CREATION_METHOD = "manual_authoring"

# ---------------------------------------------------------------------------
# Domain and language targets
# ---------------------------------------------------------------------------

DOMAIN_TARGETS = {
    "islamic_finance":    0.20,
    "gcc_banking":        0.15,
    "telecommunications": 0.15,
    "government_regulation": 0.15,
    "energy_logistics":   0.10,
    "executive_decision": 0.15,
    "arabic_english_correspondence": 0.10,
}

LANG_TARGETS = {"ar": 0.45, "en": 0.30, "ar-en": 0.25}

SPLITS = {"train": 600, "valid": 100, "eval": 150}

TASK_TYPES = [
    "question_answering", "document_summarization", "executive_briefing",
    "decision_comparison", "risk_identification", "objection_handling",
    "arabic_english_translation", "formal_correspondence",
    "regulatory_explanation", "islamic_finance_terminology",
    "structured_extraction", "refusal_missing_info",
    "hallucination_resistance", "request_for_clarification",
]

DIFFICULTIES = ["basic", "intermediate", "advanced"]

# ---------------------------------------------------------------------------
# Example templates — internally authored, no external reproduction
# ---------------------------------------------------------------------------

EXAMPLES: List[Dict[str, Any]] = [

    # ── ISLAMIC FINANCE ──────────────────────────────────────────────────────

    {"domain": "islamic_finance", "language": "ar", "task_type": "islamic_finance_terminology",
     "difficulty": "basic", "country_or_region": "GCC",
     "prompt": "ما الفرق بين عقد المرابحة وعقد الإجارة في التمويل الإسلامي؟",
     "response": (
         "المرابحة عقد بيع يُفصح فيه البائع عن تكلفة السلعة وهامش الربح، "
         "ويُستخدم عادةً لتمويل شراء الأصول. أما الإجارة فهي عقد تأجير يمنح "
         "المستأجر حق الانتفاع بالأصل مقابل أجرة دورية دون نقل الملكية. "
         "الفارق الجوهري: المرابحة تنقل الملكية، والإجارة لا تنقلها إلا باتفاق "
         "إضافي. ملاحظة: هذا توضيح مفاهيمي وليس فتوى شرعية.",
     )},

    {"domain": "islamic_finance", "language": "ar", "task_type": "question_answering",
     "difficulty": "intermediate", "country_or_region": "Kuwait",
     "prompt": "ما شروط صحة عقد المضاربة في الفقه الإسلامي؟",
     "response": (
         "تشترط صحة عقد المضاربة: (1) تحديد رأس المال نقداً أو ما يقوم مقامه، "
         "(2) اشتراط نسبة الربح لكل طرف بصورة واضحة لا مبلغاً مقطوعاً، "
         "(3) أن يكون رأس المال معلوماً ومسلَّماً للمضارب، "
         "(4) عدم تقييد المضارب بشروط تُعيق التصرف الطبيعي. "
         "الخسارة تقع على رأس المال وحده ما لم يثبت تقصير المضارب. "
         "يُنصح بمراجعة هيئة الرقابة الشرعية المختصة للتحقق من التطبيق.",
     )},

    {"domain": "islamic_finance", "language": "en", "task_type": "regulatory_explanation",
     "difficulty": "intermediate", "country_or_region": "GCC",
     "prompt": "Explain the concept of Takaful and how it differs from conventional insurance.",
     "response": (
         "Takaful is a cooperative risk-sharing arrangement grounded in the principles "
         "of mutual assistance (taawun) and donation (tabarru). Participants contribute "
         "to a shared fund; claims are paid from that fund, and any surplus may be "
         "redistributed to participants. Conventional insurance involves a commercial "
         "contract transferring risk to an insurer in exchange for a premium, which may "
         "involve elements of gharar (uncertainty) and riba (interest) considered "
         "impermissible under Islamic law. This is a conceptual explanation; specific "
         "product structures vary by jurisdiction and require Sharia board review."
     )},

    {"domain": "islamic_finance", "language": "ar-en", "task_type": "arabic_english_translation",
     "difficulty": "basic", "country_or_region": "GCC",
     "prompt": "Translate the following Islamic finance term into Arabic with a brief definition: 'Sukuk'.",
     "response": (
         "الترجمة: صكوك (جمع صَكّ).\n"
         "التعريف: الصكوك أدوات مالية إسلامية تمثّل حصصاً في ملكية أصل حقيقي "
         "أو منفعة أو خدمة، وتُدرّ عائداً مرتبطاً بأداء ذلك الأصل بدلاً من "
         "الفائدة المحظورة شرعاً. تُصدر الصكوك من قِبل حكومات وشركات في دول "
         "الخليج ودول أخرى لتمويل مشاريع متوافقة مع أحكام الشريعة الإسلامية."
     )},

    {"domain": "islamic_finance", "language": "en", "task_type": "risk_identification",
     "difficulty": "advanced", "country_or_region": "GCC",
     "prompt": "What are the key Sharia compliance risks in structuring a Murabaha facility for a corporate client?",
     "response": (
         "Key Sharia compliance risks in a Murabaha facility include: "
         "(1) Constructive ownership risk — the bank must genuinely own the asset before "
         "selling it to the client; a back-to-back arrangement where the bank never takes "
         "real ownership is impermissible. "
         "(2) Price certainty — the total deferred price must be fixed at inception; "
         "floating-rate adjustments are generally prohibited. "
         "(3) Penalty clauses — late-payment charges must not constitute riba; any "
         "penalty must be donated to charity, not retained as bank income. "
         "(4) Commodity selection — the underlying asset must be Sharia-permissible and "
         "must not be gold, silver, or currencies in most interpretations. "
         "These points require review by a qualified Sharia supervisory board; this "
         "response is informational only."
     )},

    {"domain": "islamic_finance", "language": "ar", "task_type": "document_summarization",
     "difficulty": "intermediate", "country_or_region": "Kuwait",
     "prompt": (
         "لخّص المبادئ الأساسية لمعيار هيئة المحاسبة والمراجعة للمؤسسات المالية "
         "الإسلامية (أيوفي) المتعلق بالمشاركة المتناقصة."
     ),
     "response": (
         "تُعرِّف معايير أيوفي المشاركة المتناقصة بوصفها شراكة يتملّك فيها "
         "العميل تدريجياً حصة المؤسسة المالية عبر دفعات دورية، مع دفع أجرة "
         "على الحصة المتبقية للمؤسسة. المبادئ الجوهرية: (1) وجوب تحديد حصص "
         "الملكية بدقة، (2) استقلالية عقد الإجارة عن عقد الشراء، (3) عدم "
         "إلزام العميل مسبقاً بشراء الحصص. يُرجى الرجوع إلى النص الكامل "
         "للمعيار والهيئة الشرعية المختصة للتطبيق الفعلي."
     )},

    {"domain": "islamic_finance", "language": "en", "task_type": "executive_briefing",
     "difficulty": "advanced", "country_or_region": "GCC",
     "prompt": "Prepare a one-paragraph executive briefing on the growth of the GCC sukuk market.",
     "response": (
         "The GCC sukuk market has expanded significantly over the past decade, driven "
         "by sovereign issuances from Saudi Arabia, the UAE, and Kuwait, as well as "
         "growing corporate demand for Sharia-compliant capital. The market provides "
         "governments with an alternative to conventional bonds for infrastructure "
         "financing, while offering institutional investors fixed-income-like instruments "
         "aligned with Islamic principles. Regulatory harmonisation across GCC central "
         "banks and the adoption of AAOIFI standards have improved cross-border "
         "issuance. Accurate market-size figures require reference to current data from "
         "the Islamic Financial Services Board (IFSB) or Bloomberg; this briefing "
         "reflects general publicly available trends and should not be cited as a "
         "primary source."
     )},

    {"domain": "islamic_finance", "language": "ar", "task_type": "hallucination_resistance",
     "difficulty": "advanced", "country_or_region": "GCC",
     "prompt": "ما هو سعر الفائدة الذي تفرضه بنوك الخليج على عقود المرابحة حالياً؟",
     "response": (
         "لا تتضمن عقود المرابحة فائدة بالمفهوم التقليدي، بل هامش ربح متفق عليه "
         "مسبقاً. تتفاوت هوامش الربح بحسب المؤسسة والمنتج وظروف السوق، ولا "
         "يمكنني تقديم أرقام محددة وحديثة دون الرجوع إلى البيانات الرسمية "
         "للبنوك أو البنوك المركزية المعنية. يُرجى التواصل مع المؤسسة المالية "
         "مباشرةً للحصول على عروض الأسعار الحالية."
     )},

    {"domain": "islamic_finance", "language": "en", "task_type": "decision_comparison",
     "difficulty": "intermediate", "country_or_region": "GCC",
     "prompt": "Compare Mudaraba and Musharaka as financing structures for a GCC startup.",
     "response": (
         "In a Mudaraba arrangement, one party (rab al-mal) provides capital while the "
         "other (mudarib) provides expertise and management; profits are shared by "
         "pre-agreed ratio, but financial losses fall entirely on the capital provider "
         "unless the mudarib is negligent. In a Musharaka, both parties contribute "
         "capital and share profits and losses proportionally to their stakes. For a "
         "GCC startup, Musharaka may be preferred when the investor wishes to retain "
         "governance rights and share operational risk, while Mudaraba suits a "
         "passive investor backing a skilled entrepreneur. Both structures require "
         "Sharia board approval and careful documentation of profit-sharing ratios "
         "before execution."
     )},

    {"domain": "islamic_finance", "language": "ar", "task_type": "formal_correspondence",
     "difficulty": "intermediate", "country_or_region": "Kuwait",
     "prompt": "اكتب رسالة رسمية موجزة تطلب فيها من بنك إسلامي الحصول على عرض تمويل مرابحة لشراء معدات صناعية.",
     "response": (
         "السادة / إدارة التمويل الإسلامي\n"
         "تحية طيبة وبعد،\n\n"
         "يسرّنا التواصل معكم بشأن الحصول على عرض تمويل بصيغة المرابحة لشراء "
         "معدات صناعية تُقدَّر قيمتها الإجمالية بـ [المبلغ]. نودّ الاطلاع على "
         "شروط التمويل المتاحة، وهامش الربح المعتمد، ومدة السداد المقترحة، "
         "فضلاً عن المستندات المطلوبة لاستكمال الإجراءات.\n\n"
         "يُرجى إرسال العرض إلى عنوان البريد الإلكتروني المدرج أدناه في أقرب "
         "وقت ممكن.\n\n"
         "مع خالص التقدير،\n[الاسم] — [المسمى الوظيفي] — [الشركة]"
     )},

    {"domain": "islamic_finance", "language": "ar-en", "task_type": "structured_extraction",
     "difficulty": "intermediate", "country_or_region": "GCC",
     "prompt": (
         "Extract the key financial terms from the following Murabaha term sheet summary "
         "and present them in a structured table:\n"
         "The bank agrees to sell Asset X to the client at a deferred price of USD 1,200,000, "
         "representing the cost of USD 1,000,000 plus a profit margin of 20%. "
         "The deferred price is payable in 24 equal monthly instalments. "
         "Late payment will result in a charitable donation of 1% per annum on the overdue amount."
     ),
     "response": (
         "| Term | Value |\n"
         "|---|---|\n"
         "| Asset cost (ثمن التكلفة) | USD 1,000,000 |\n"
         "| Profit margin (هامش الربح) | 20% |\n"
         "| Deferred sale price (الثمن الآجل) | USD 1,200,000 |\n"
         "| Repayment period | 24 months |\n"
         "| Instalment amount | USD 50,000/month |\n"
         "| Late payment penalty | 1% p.a. on overdue amount (donated to charity) |"
     )},

    {"domain": "islamic_finance", "language": "en", "task_type": "refusal_missing_info",
     "difficulty": "basic", "country_or_region": "GCC",
     "prompt": "Is the investment product offered by XYZ Islamic Bank Sharia-compliant?",
     "response": (
         "I cannot determine the Sharia compliance of a specific product without "
         "reviewing its full terms, structure, and the opinion of a qualified Sharia "
         "supervisory board. Compliance depends on the specific contractual arrangements, "
         "the underlying assets, and the applicable scholarly interpretation. Please "
         "request the Sharia certificate and the board's fatwa directly from XYZ Islamic "
         "Bank for an authoritative answer."
     )},

    # ── GCC BANKING ──────────────────────────────────────────────────────────

    {"domain": "gcc_banking", "language": "en", "task_type": "regulatory_explanation",
     "difficulty": "intermediate", "country_or_region": "Kuwait",
     "prompt": "What are the primary Basel III capital adequacy requirements applicable to Kuwaiti banks?",
     "response": (
         "Under Basel III, as implemented by the Central Bank of Kuwait, banks are "
         "required to maintain: (1) a Common Equity Tier 1 (CET1) ratio of at least "
         "4.5% of risk-weighted assets; (2) a Tier 1 capital ratio of at least 6%; "
         "(3) a total capital ratio of at least 8%; and (4) a capital conservation "
         "buffer of 2.5%, bringing the effective CET1 requirement to 7%. The CBK may "
         "impose additional requirements for systemically important institutions. "
         "Exact current thresholds should be verified against the latest CBK circulars, "
         "as regulatory requirements are subject to revision."
     )},

    {"domain": "gcc_banking", "language": "ar", "task_type": "question_answering",
     "difficulty": "basic", "country_or_region": "Kuwait",
     "prompt": "ما الفرق بين رأس المال من الشريحة الأولى والشريحة الثانية في إطار بازل؟",
     "response": (
         "رأس المال من الشريحة الأولى (Tier 1) هو رأس المال الأساسي للبنك، "
         "ويشمل حقوق المساهمين العادية والأرباح المحتجزة. يُعدّ الأكثر قدرة "
         "على استيعاب الخسائر. أما رأس المال من الشريحة الثانية (Tier 2) "
         "فيشمل الديون الثانوية وبعض الاحتياطيات، وهو أقل قدرة على استيعاب "
         "الخسائر مقارنةً بالشريحة الأولى. تُحدد نسب كل شريحة بموجب متطلبات "
         "بازل الصادرة عن البنوك المركزية المختصة."
     )},

    {"domain": "gcc_banking", "language": "en", "task_type": "risk_identification",
     "difficulty": "advanced", "country_or_region": "GCC",
     "prompt": "Identify the key credit risks in extending a syndicated loan to a GCC petrochemical company.",
     "response": (
         "Key credit risks include: (1) Commodity price risk — revenue is tied to "
         "hydrocarbon and petrochemical prices, which are cyclical and volatile. "
         "(2) Offtake concentration — dependence on a small number of buyers increases "
         "counterparty risk. (3) Regulatory and environmental risk — evolving GCC "
         "environmental regulations may impose capital expenditure obligations. "
         "(4) Refinancing risk — large syndicated facilities may face rollover challenges "
         "in tightening credit markets. (5) FX risk — if revenues are in USD but costs "
         "include local-currency components, margin compression is possible. "
         "A thorough credit assessment requires current audited financials and "
         "independent legal review of the facility agreement."
     )},

    {"domain": "gcc_banking", "language": "ar", "task_type": "document_summarization",
     "difficulty": "intermediate", "country_or_region": "UAE",
     "prompt": "لخّص المتطلبات الرئيسية لمكافحة غسل الأموال المفروضة على البنوك الإماراتية.",
     "response": (
         "تُلزم المتطلبات الرئيسية لمكافحة غسل الأموال في الإمارات البنوكَ بما يلي: "
         "(1) تطبيق إجراءات العناية الواجبة بالعميل (KYC) عند فتح الحسابات وبصفة دورية، "
         "(2) مراقبة المعاملات والإبلاغ عن المعاملات المشبوهة لوحدة المعلومات المالية، "
         "(3) الاحتفاظ بسجلات المعاملات لمدة لا تقل عن خمس سنوات، "
         "(4) تعيين مسؤول امتثال مؤهل. تستند هذه المتطلبات إلى قانون مكافحة غسل الأموال "
         "الإماراتي والتعليمات الصادرة عن المصرف المركزي. يُنصح بمراجعة أحدث التعاميم "
         "للتحقق من المتطلبات الحالية."
     )},

    {"domain": "gcc_banking", "language": "en", "task_type": "executive_briefing",
     "difficulty": "intermediate", "country_or_region": "GCC",
     "prompt": "Provide a one-paragraph executive briefing on the role of GCC central banks in managing liquidity.",
     "response": (
         "GCC central banks — including the Central Bank of Kuwait, Saudi Central Bank "
         "(SAMA), and the UAE Central Bank — manage domestic liquidity primarily through "
         "open market operations, reserve requirements, and standing lending facilities. "
         "Given that most GCC currencies are pegged to the US dollar, monetary policy "
         "transmission is constrained, making liquidity management a critical tool for "
         "financial stability. During periods of low oil revenues, central banks have "
         "drawn on sovereign wealth fund deposits and issued domestic debt instruments "
         "to support banking sector liquidity. Specific current policy rates and "
         "reserve requirements should be verified against official central bank "
         "publications."
     )},

    {"domain": "gcc_banking", "language": "ar-en", "task_type": "formal_correspondence",
     "difficulty": "intermediate", "country_or_region": "Kuwait",
     "prompt": "Draft a formal letter in Arabic requesting a credit facility review from a Kuwaiti commercial bank.",
     "response": (
         "السادة / إدارة الائتمان\n"
         "تحية طيبة وبعد،\n\n"
         "نتشرف بتقديم طلبنا لمراجعة التسهيلات الائتمانية الممنوحة لشركتنا، "
         "وذلك في ضوء التطورات الأخيرة في أعمالنا وحجم المعاملات المصرفية. "
         "نأمل أن تأخذوا بعين الاعتبار السجل الائتماني الجيد لشركتنا وحجم "
         "الأعمال المشتركة، ونطلب تحديد موعد لمناقشة إمكانية تعديل حدود "
         "التسهيلات الحالية.\n\n"
         "نحن على استعداد لتقديم أي مستندات مالية إضافية عند الطلب.\n\n"
         "مع خالص التقدير،\n[الاسم] — [المسمى الوظيفي]"
     )},

    {"domain": "gcc_banking", "language": "en", "task_type": "hallucination_resistance",
     "difficulty": "basic", "country_or_region": "Kuwait",
     "prompt": "What is the current interest rate set by the Central Bank of Kuwait?",
     "response": (
         "I do not have access to real-time data and cannot provide the current "
         "discount rate set by the Central Bank of Kuwait. Rates are subject to "
         "change and should be verified directly on the CBK's official website "
         "(www.cbk.gov.kw) or through a licensed financial data provider."
     )},

    {"domain": "gcc_banking", "language": "ar", "task_type": "structured_extraction",
     "difficulty": "intermediate", "country_or_region": "GCC",
     "prompt": (
         "استخرج المعلومات الرئيسية من الملخص التالي لتقرير ائتماني:\n"
         "الشركة: مجموعة الخليج للصناعات. إجمالي الأصول: 450 مليون دولار. "
         "نسبة الدين إلى حقوق الملكية: 1.8. التصنيف الائتماني: BBB. "
         "صافي الربح للسنة المالية الأخيرة: 22 مليون دولار."
     ),
     "response": (
         "| البند | القيمة |\n"
         "|---|---|\n"
         "| اسم الشركة | مجموعة الخليج للصناعات |\n"
         "| إجمالي الأصول | 450 مليون دولار |\n"
         "| نسبة الدين إلى حقوق الملكية | 1.8 |\n"
         "| التصنيف الائتماني | BBB |\n"
         "| صافي الربح | 22 مليون دولار |"
     )},

    # ── TELECOMMUNICATIONS ───────────────────────────────────────────────────

    {"domain": "telecommunications", "language": "en", "task_type": "regulatory_explanation",
     "difficulty": "intermediate", "country_or_region": "Kuwait",
     "prompt": "What are the key regulatory obligations for mobile network operators in Kuwait?",
     "response": (
         "Mobile network operators in Kuwait are regulated by the Communications and "
         "Information Technology Regulatory Authority (CITRA). Key obligations include: "
         "(1) obtaining and maintaining valid spectrum licences; (2) meeting quality-of-"
         "service (QoS) benchmarks for voice and data services; (3) complying with "
         "lawful interception requirements; (4) adhering to number portability "
         "regulations; and (5) meeting local content and investment commitments where "
         "applicable. Specific current requirements should be verified against CITRA's "
         "published regulations and licence conditions, as these are subject to revision."
     )},

    {"domain": "telecommunications", "language": "ar", "task_type": "question_answering",
     "difficulty": "basic", "country_or_region": "GCC",
     "prompt": "ما المقصود بجودة الخدمة (QoS) في شبكات الاتصالات؟",
     "response": (
         "جودة الخدمة (QoS) هي مجموعة من المعايير التقنية التي تضمن أداء "
         "الشبكة وفق مستويات محددة، وتشمل: زمن الاستجابة (Latency)، "
         "ومعدل فقدان الحزم (Packet Loss)، وعرض النطاق الترددي (Bandwidth)، "
         "والتذبذب (Jitter). تُحدد هيئات الاتصالات في دول الخليج حدوداً دنيا "
         "لهذه المعايير يلتزم بها مشغلو الشبكات لضمان تجربة مستخدم مقبولة."
     )},

    {"domain": "telecommunications", "language": "en", "task_type": "risk_identification",
     "difficulty": "advanced", "country_or_region": "GCC",
     "prompt": "Identify the main commercial risks for a GCC telecom operator expanding into enterprise managed services.",
     "response": (
         "Key commercial risks include: (1) Margin compression — enterprise managed "
         "services require significant upfront investment in talent and infrastructure "
         "with longer sales cycles than consumer services. (2) Integration complexity — "
         "connecting legacy telecom infrastructure with modern cloud-native enterprise "
         "platforms creates technical and operational risk. (3) Competition from global "
         "system integrators — established players such as Accenture, IBM, and "
         "hyperscalers may outbid on large contracts. (4) Talent scarcity — skilled "
         "enterprise IT professionals are in short supply across the GCC. (5) Contract "
         "concentration — dependence on a small number of government or energy-sector "
         "clients creates revenue volatility. These are general risk categories; "
         "specific assessments require current market data."
     )},

    {"domain": "telecommunications", "language": "ar", "task_type": "executive_briefing",
     "difficulty": "intermediate", "country_or_region": "Kuwait",
     "prompt": "أعدّ موجزاً تنفيذياً موجزاً حول تأثير تقنية الجيل الخامس على قطاع الاتصالات في الكويت.",
     "response": (
         "تُتيح تقنية الجيل الخامس (5G) في الكويت فرصاً لتعزيز الاتصال بين "
         "الأجهزة (IoT) ودعم التحول الرقمي في القطاعين الحكومي والصناعي. "
         "تواجه شركات الاتصالات الكويتية تحديات تتعلق بتكاليف ترقية البنية "
         "التحتية وتوفر الطيف الترددي المناسب. يتوقع المشغلون تحقيق عائد "
         "استثماري من خلال خدمات الشركات والمدن الذكية، وإن كان التوقيت الفعلي "
         "يعتمد على معدلات الاعتماد والإطار التنظيمي. تستند هذه المعلومات إلى "
         "توجهات عامة ولا تعكس بيانات تشغيلية محددة."
     )},

    {"domain": "telecommunications", "language": "en", "task_type": "decision_comparison",
     "difficulty": "advanced", "country_or_region": "GCC",
     "prompt": "Compare the build-versus-buy decision for a GCC telecom operator considering a cloud-native BSS platform.",
     "response": (
         "Building a proprietary BSS platform offers maximum customisation and long-term "
         "IP ownership but requires significant upfront investment, extended delivery "
         "timelines (typically 3–5 years), and sustained internal engineering capability. "
         "Buying a commercial off-the-shelf (COTS) solution from vendors such as Amdocs "
         "or Ericsson accelerates time-to-market and transfers maintenance risk, but "
         "introduces vendor dependency and may require costly customisation for GCC-"
         "specific regulatory and Arabic-language requirements. A hybrid approach — "
         "adopting a COTS core with custom Arabic-language and regulatory modules — "
         "is increasingly common in the region. The optimal choice depends on the "
         "operator's existing technical debt, budget cycle, and strategic roadmap; "
         "a formal TCO analysis is recommended before commitment."
     )},

    {"domain": "telecommunications", "language": "ar-en", "task_type": "formal_correspondence",
     "difficulty": "basic", "country_or_region": "Kuwait",
     "prompt": "Write a short formal email in Arabic to a telecom vendor requesting a product demonstration.",
     "response": (
         "الموضوع: طلب عرض توضيحي لمنتج [اسم المنتج]\n\n"
         "السادة / فريق المبيعات\n"
         "تحية طيبة وبعد،\n\n"
         "يسرّنا التواصل معكم للاستفسار عن إمكانية ترتيب عرض توضيحي لمنتجكم "
         "[اسم المنتج]، وذلك لتقييم مدى ملاءمته لاحتياجات شركتنا التشغيلية. "
         "نرجو إفادتنا بالمواعيد المتاحة خلال الأسبوعين القادمين.\n\n"
         "مع خالص التقدير،\n[الاسم] — [القسم]"
     )},

    {"domain": "telecommunications", "language": "en", "task_type": "hallucination_resistance",
     "difficulty": "intermediate", "country_or_region": "Kuwait",
     "prompt": "What is the exact 5G coverage percentage in Kuwait as of today?",
     "response": (
         "I do not have access to real-time network coverage data and cannot provide "
         "a current 5G coverage percentage for Kuwait. Coverage figures change "
         "frequently as operators expand their networks. For accurate, up-to-date "
         "information, please consult the official websites of Zain Kuwait, Ooredoo "
         "Kuwait, and STC Kuwait, or the Communications and Information Technology "
         "Regulatory Authority (CITRA)."
     )},

    # ── GOVERNMENT AND REGULATION ────────────────────────────────────────────

    {"domain": "government_regulation", "language": "ar", "task_type": "regulatory_explanation",
     "difficulty": "intermediate", "country_or_region": "Kuwait",
     "prompt": "ما الإطار التنظيمي الذي يحكم إجراءات المشتريات الحكومية في الكويت؟",
     "response": (
         "تخضع المشتريات الحكومية في الكويت بصفة رئيسية لأحكام قانون المناقصات "
         "العامة وتعديلاته، الذي تُشرف على تطبيقه الجهاز المركزي للمناقصات "
         "العامة. يُلزم القانون الجهات الحكومية بطرح المناقصات العامة للعقود "
         "التي تتجاوز حدوداً مالية محددة، مع منح أولوية للمنتجات والشركات "
         "الكويتية في بعض الحالات. يُنصح بمراجعة النص الرسمي للقانون والتعاميم "
         "الصادرة عن الجهاز للاطلاع على الإجراءات التفصيلية المحدّثة."
     )},

    {"domain": "government_regulation", "language": "en", "task_type": "document_summarization",
     "difficulty": "intermediate", "country_or_region": "GCC",
     "prompt": "Summarise the key principles of GCC data protection frameworks.",
     "response": (
         "GCC data protection frameworks vary by jurisdiction but share several common "
         "principles: (1) lawful basis for processing personal data; (2) purpose "
         "limitation — data collected for a specific purpose must not be used for "
         "incompatible purposes; (3) data minimisation — only necessary data should "
         "be collected; (4) data subject rights including access, correction, and in "
         "some jurisdictions deletion; (5) cross-border transfer restrictions requiring "
         "adequate protection in the recipient country. Saudi Arabia's PDPL, the UAE's "
         "Federal DPL, and Kuwait's draft data protection law each have specific "
         "provisions. Legal counsel should be consulted for jurisdiction-specific "
         "compliance obligations."
     )},

    {"domain": "government_regulation", "language": "ar", "task_type": "question_answering",
     "difficulty": "basic", "country_or_region": "Kuwait",
     "prompt": "ما دور ديوان المحاسبة في الكويت؟",
     "response": (
         "ديوان المحاسبة في الكويت هو الجهاز الرقابي المالي الأعلى في الدولة، "
         "ويختص بمراجعة الحسابات الختامية للجهات الحكومية والتحقق من صحة "
         "الإنفاق العام وفق الأنظمة والقوانين المعمول بها. يرفع الديوان تقاريره "
         "إلى مجلس الأمة ويُسهم في تعزيز الشفافية والمساءلة في إدارة المال "
         "العام. يستند هذا الوصف إلى المعلومات المتاحة للعموم ولا يُعدّ تفسيراً "
         "قانونياً رسمياً."
     )},

    {"domain": "government_regulation", "language": "en", "task_type": "risk_identification",
     "difficulty": "advanced", "country_or_region": "GCC",
     "prompt": "What are the key regulatory risks for a foreign company establishing a branch in Kuwait?",
     "response": (
         "Key regulatory risks include: (1) Commercial agency requirements — certain "
         "activities may require a Kuwaiti commercial agent, limiting operational "
         "independence. (2) Foreign ownership restrictions — some sectors restrict "
         "foreign equity participation; the Foreign Direct Investment Law provides "
         "exemptions in specific cases. (3) Labour nationalisation (Kuwaitisation) "
         "quotas — companies must meet prescribed ratios of Kuwaiti employees. "
         "(4) Regulatory approval timelines — licensing processes can be lengthy and "
         "subject to change. (5) Dispute resolution — commercial disputes may be "
         "subject to Kuwaiti courts unless arbitration is contractually agreed. "
         "Legal counsel familiar with Kuwaiti commercial law should be engaged before "
         "market entry."
     )},

    {"domain": "government_regulation", "language": "ar", "task_type": "executive_briefing",
     "difficulty": "intermediate", "country_or_region": "GCC",
     "prompt": "أعدّ موجزاً تنفيذياً حول أثر رؤية 2030 السعودية على بيئة الأعمال في المنطقة.",
     "response": (
         "تسعى رؤية المملكة العربية السعودية 2030 إلى تنويع الاقتصاد الوطني "
         "وتقليل الاعتماد على النفط، من خلال تطوير قطاعات السياحة والترفيه "
         "والصناعة والخدمات المالية. أسهمت الإصلاحات التنظيمية المصاحبة "
         "في تحسين مناخ الأعمال وجذب الاستثمار الأجنبي المباشر. تنعكس هذه "
         "التحولات على دول الخليج المجاورة من خلال تصاعد التنافسية الإقليمية "
         "وفرص التكامل الاقتصادي. تستند هذه المعلومات إلى مصادر عامة متاحة "
         "وتعكس التوجهات العامة لا الأرقام الرسمية الحديثة."
     )},

    {"domain": "government_regulation", "language": "ar-en", "task_type": "arabic_english_translation",
     "difficulty": "intermediate", "country_or_region": "Kuwait",
     "prompt": "Translate the following regulatory term into Arabic with a brief explanation: 'Regulatory Sandbox'.",
     "response": (
         "الترجمة: البيئة التنظيمية التجريبية (أو الحاضنة التنظيمية).\n"
         "الشرح: هي إطار تنظيمي يُتيح للشركات الناشئة والمبتكرين اختبار "
         "منتجاتهم وخدماتهم المالية أو التقنية في بيئة محكومة تحت إشراف "
         "الجهة التنظيمية، مع تخفيف بعض الاشتراطات التنظيمية المعتادة "
         "لفترة محددة. تعتمد عدة دول خليجية هذا النهج لتحفيز الابتكار في "
         "قطاع التكنولوجيا المالية."
     )},

    {"domain": "government_regulation", "language": "en", "task_type": "refusal_missing_info",
     "difficulty": "basic", "country_or_region": "Kuwait",
     "prompt": "What is the exact penalty for a company that violates Kuwait's data protection law?",
     "response": (
         "Kuwait's comprehensive data protection law was still in draft form as of my "
         "knowledge cutoff; I cannot provide exact penalty figures for a law that may "
         "not yet be fully enacted. For authoritative and current information, please "
         "consult the Communications and Information Technology Regulatory Authority "
         "(CITRA) or a licensed Kuwaiti legal practitioner."
     )},

    # ── ENERGY AND LOGISTICS ─────────────────────────────────────────────────

    {"domain": "energy_logistics", "language": "en", "task_type": "executive_briefing",
     "difficulty": "intermediate", "country_or_region": "GCC",
     "prompt": "Provide a brief executive briefing on the strategic importance of the GCC's LNG export capacity.",
     "response": (
         "The GCC region, led by Qatar as one of the world's largest LNG exporters, "
         "plays a critical role in global energy security. LNG exports provide "
         "long-term revenue diversification beyond crude oil and support the region's "
         "geopolitical influence. Ongoing capacity expansions — including Qatar's "
         "North Field expansion — are expected to increase global LNG supply "
         "significantly over the coming decade. For GCC governments, LNG revenue "
         "supports sovereign wealth fund accumulation and domestic economic development. "
         "Specific production volumes and capacity figures should be verified against "
         "current data from the International Energy Agency or Qatar Energy."
     )},

    {"domain": "energy_logistics", "language": "ar", "task_type": "question_answering",
     "difficulty": "basic", "country_or_region": "Kuwait",
     "prompt": "ما دور مؤسسة البترول الكويتية في قطاع الطاقة؟",
     "response": (
         "مؤسسة البترول الكويتية (KPC) هي الشركة القابضة الحكومية المسؤولة عن "
         "إدارة قطاع النفط والغاز في الكويت، وتشمل عملياتها الاستكشاف والإنتاج "
         "والتكرير والتوزيع والتسويق الدولي عبر شركاتها التابعة. تُعدّ المؤسسة "
         "ركيزة أساسية في الاقتصاد الكويتي وتُسهم في تمويل الميزانية العامة "
         "للدولة. تستند هذه المعلومات إلى البيانات المتاحة للعموم ولا تعكس "
         "أرقاماً تشغيلية حديثة."
     )},

    {"domain": "energy_logistics", "language": "en", "task_type": "risk_identification",
     "difficulty": "advanced", "country_or_region": "GCC",
     "prompt": "What are the primary supply chain risks for a GCC petrochemical company exporting to Asian markets?",
     "response": (
         "Primary supply chain risks include: (1) Shipping lane disruption — "
         "geopolitical tensions in the Strait of Hormuz or the Red Sea can delay "
         "or divert tanker routes, increasing freight costs. (2) Demand volatility — "
         "Asian petrochemical demand is sensitive to Chinese economic cycles and "
         "local capacity additions. (3) Port congestion — major Asian ports "
         "periodically experience congestion, extending delivery lead times. "
         "(4) Feedstock price risk — naphtha and ethane price movements affect "
         "production economics and competitiveness. (5) Regulatory risk — "
         "evolving environmental standards in importing countries may require "
         "product reformulation. Risk mitigation typically involves diversified "
         "customer bases, long-term offtake agreements, and hedging strategies."
     )},

    {"domain": "energy_logistics", "language": "ar", "task_type": "document_summarization",
     "difficulty": "intermediate", "country_or_region": "GCC",
     "prompt": "لخّص أهمية الموانئ الخليجية في سلاسل الإمداد الإقليمية.",
     "response": (
         "تُشكّل الموانئ الخليجية الكبرى — كميناء جبل علي في دبي وميناء الملك "
         "عبدالله في السعودية وميناء الشويخ في الكويت — عقداً محورية في سلاسل "
         "الإمداد الإقليمية والدولية. توفر هذه الموانئ بنية تحتية لوجستية "
         "متكاملة تشمل المناطق الحرة ومرافق التخزين والتوزيع، مما يُعزز مكانة "
         "الخليج مركزاً للتجارة بين آسيا وأفريقيا وأوروبا. تستند هذه المعلومات "
         "إلى بيانات متاحة للعموم وقد لا تعكس أحدث أرقام الطاقة الاستيعابية."
     )},

    {"domain": "energy_logistics", "language": "en", "task_type": "decision_comparison",
     "difficulty": "intermediate", "country_or_region": "GCC",
     "prompt": "Compare rail and road freight for intra-GCC logistics from a cost and reliability perspective.",
     "response": (
         "Road freight currently dominates intra-GCC logistics due to its flexibility, "
         "door-to-door capability, and the extensive highway network connecting GCC "
         "states. However, it is subject to border crossing delays, driver shortages, "
         "and fuel cost volatility. Rail freight, once the GCC Railway Network is "
         "fully operational, is expected to offer lower per-tonne costs for bulk "
         "commodities over long distances and greater reliability for scheduled "
         "services. Rail is less suited to time-sensitive or fragmented shipments. "
         "The optimal mode depends on cargo type, volume, urgency, and origin-"
         "destination pair. The GCC Railway project timeline should be verified "
         "against current official announcements."
     )},

    # ── EXECUTIVE DECISION ANALYSIS ──────────────────────────────────────────

    {"domain": "executive_decision", "language": "en", "task_type": "decision_comparison",
     "difficulty": "advanced", "country_or_region": "GCC",
     "prompt": "Compare the strategic trade-offs between organic growth and acquisition for a GCC financial services firm.",
     "response": (
         "Organic growth preserves cultural coherence and avoids integration risk, "
         "but is slower and may cede market share to more aggressive competitors. "
         "It is typically preferred when the firm's core capabilities are a genuine "
         "competitive advantage and when target markets are accessible through "
         "existing distribution. Acquisition accelerates market entry and capability "
         "acquisition but introduces integration complexity, potential cultural "
         "misalignment, and valuation risk — particularly in GCC markets where "
         "pricing can be elevated due to limited supply of quality targets. "
         "A rigorous due diligence process, clear synergy quantification, and a "
         "defined integration plan are prerequisites for a successful acquisition. "
         "The optimal path depends on the firm's capital position, risk appetite, "
         "and strategic timeline."
     )},

    {"domain": "executive_decision", "language": "ar", "task_type": "executive_briefing",
     "difficulty": "intermediate", "country_or_region": "Kuwait",
     "prompt": "أعدّ موجزاً تنفيذياً حول اعتبارات التحول الرقمي للمؤسسات الحكومية الكويتية.",
     "response": (
         "يُمثّل التحول الرقمي للمؤسسات الحكومية الكويتية أولوية استراتيجية "
         "تتضمن رقمنة الخدمات الحكومية وتحديث البنية التحتية لتقنية المعلومات "
         "وتعزيز قدرات الأمن السيبراني. تبرز تحديات رئيسية تشمل إدارة التغيير "
         "المؤسسي وتوافر الكوادر البشرية المؤهلة وضمان التكامل بين الأنظمة "
         "القائمة والحلول الجديدة. تُسهم مبادرات مثل الحكومة الإلكترونية في "
         "تحسين كفاءة الخدمات وتقليل التكاليف التشغيلية على المدى البعيد. "
         "تستند هذه المعلومات إلى توجهات عامة ولا تعكس خطط حكومية رسمية محددة."
     )},

    {"domain": "executive_decision", "language": "en", "task_type": "risk_identification",
     "difficulty": "advanced", "country_or_region": "GCC",
     "prompt": "What are the top five strategic risks facing GCC family businesses in the next decade?",
     "response": (
         "The top five strategic risks include: (1) Succession and governance — "
         "transitioning leadership across generations without formalised governance "
         "structures is a leading cause of value destruction. (2) Digital disruption — "
         "established family businesses in retail, real estate, and services face "
         "competition from digitally native entrants. (3) Talent retention — "
         "attracting and retaining non-family professional management is increasingly "
         "challenging as the regional talent market deepens. (4) Capital allocation "
         "discipline — without institutional oversight, capital may be deployed "
         "suboptimally across family interests. (5) Regulatory change — "
         "evolving corporate governance and transparency requirements across GCC "
         "jurisdictions may require structural adjustments. These are general risk "
         "categories; specific assessments require detailed knowledge of the "
         "individual business."
     )},

    {"domain": "executive_decision", "language": "ar", "task_type": "objection_handling",
     "difficulty": "intermediate", "country_or_region": "GCC",
     "prompt": "كيف يمكن الرد على اعتراض أحد أعضاء مجلس الإدارة بأن الاستثمار في الذكاء الاصطناعي مكلف للغاية؟",
     "response": (
         "يمكن معالجة هذا الاعتراض من خلال عدة محاور: أولاً، التمييز بين "
         "تكاليف البنية التحتية الضخمة لتدريب النماذج الكبرى وتكاليف تبني "
         "حلول الذكاء الاصطناعي الجاهزة أو المُخصَّصة التي تكون أكثر اقتصادية. "
         "ثانياً، تقديم حالات استخدام محددة مع تقدير عائد الاستثمار المتوقع "
         "وفق معطيات الشركة. ثالثاً، اقتراح نهج تدريجي يبدأ بمشاريع تجريبية "
         "محدودة التكلفة لإثبات الجدوى قبل التوسع. رابعاً، الإشارة إلى تكلفة "
         "عدم الاستثمار في ظل تنامي اعتماد المنافسين على هذه التقنيات."
     )},

    {"domain": "executive_decision", "language": "en", "task_type": "request_for_clarification",
     "difficulty": "basic", "country_or_region": "GCC",
     "prompt": "Should our company invest in the GCC market?",
     "response": (
         "To provide a meaningful analysis, I would need additional information: "
         "(1) What is your company's industry and current geographic footprint? "
         "(2) Which specific GCC country or countries are you considering? "
         "(3) What is the investment horizon and capital available? "
         "(4) Are you considering organic entry, partnership, or acquisition? "
         "(5) What is the primary strategic objective — revenue growth, cost "
         "optimisation, or market diversification? "
         "With these details, I can offer a more targeted assessment of the "
         "opportunity and associated risks."
     )},

    {"domain": "executive_decision", "language": "ar-en", "task_type": "structured_extraction",
     "difficulty": "intermediate", "country_or_region": "GCC",
     "prompt": (
         "Extract the key decision criteria from the following executive summary:\n"
         "The board is evaluating three options: (A) expand domestically with a "
         "USD 50M investment, expected ROI 18% over 5 years; (B) enter the Saudi "
         "market via a joint venture, USD 30M investment, expected ROI 22% over "
         "5 years, higher execution risk; (C) defer all expansion for 2 years "
         "pending regulatory clarity."
     ),
     "response": (
         "| Option | Investment | Expected ROI | Timeline | Key Risk |\n"
         "|---|---|---|---|---|\n"
         "| A — Domestic expansion | USD 50M | 18% | 5 years | Market saturation |\n"
         "| B — Saudi JV | USD 30M | 22% | 5 years | Execution / partner risk |\n"
         "| C — Defer | USD 0 | N/A | 2-year delay | Opportunity cost |"
     )},

    {"domain": "executive_decision", "language": "en", "task_type": "hallucination_resistance",
     "difficulty": "intermediate", "country_or_region": "GCC",
     "prompt": "What was the exact GDP growth rate of Kuwait in 2024?",
     "response": (
         "I do not have access to verified final GDP figures for Kuwait for 2024. "
         "Economic statistics are revised after initial publication and my training "
         "data may not reflect the most recent official releases. For authoritative "
         "figures, please consult the Central Statistical Bureau of Kuwait, the "
         "International Monetary Fund's World Economic Outlook database, or the "
         "World Bank's data portal."
     )},

    # ── ARABIC-ENGLISH CORRESPONDENCE ────────────────────────────────────────

    {"domain": "arabic_english_correspondence", "language": "ar", "task_type": "formal_correspondence",
     "difficulty": "basic", "country_or_region": "Kuwait",
     "prompt": "اكتب رسالة رسمية باللغة العربية لتأكيد حضور اجتماع عمل.",
     "response": (
         "الموضوع: تأكيد حضور الاجتماع\n\n"
         "السادة / [اسم الجهة]\n"
         "تحية طيبة وبعد،\n\n"
         "يسعدنا تأكيد حضورنا للاجتماع المقرر بتاريخ [التاريخ] في تمام الساعة "
         "[الوقت] في [المكان]. سيمثّل شركتنا في هذا الاجتماع كل من [الاسم الأول] "
         "و[الاسم الثاني].\n\n"
         "نتطلع إلى لقاء فريقكم ومناقشة المحاور المدرجة في جدول الأعمال.\n\n"
         "مع خالص التقدير،\n[الاسم] — [المسمى الوظيفي]"
     )},

    {"domain": "arabic_english_correspondence", "language": "en", "task_type": "formal_correspondence",
     "difficulty": "basic", "country_or_region": "GCC",
     "prompt": "Write a formal English email declining a business proposal politely.",
     "response": (
         "Subject: RE: Business Proposal — [Proposal Reference]\n\n"
         "Dear [Name],\n\n"
         "Thank you for presenting your proposal and for the time you invested in "
         "preparing it. After careful consideration, we have decided not to proceed "
         "with this opportunity at this time. This decision reflects our current "
         "strategic priorities rather than any deficiency in your proposal.\n\n"
         "We appreciate your interest in working with us and hope there may be "
         "opportunities to collaborate in the future.\n\n"
         "Kind regards,\n[Name] — [Title]"
     )},

    {"domain": "arabic_english_correspondence", "language": "ar-en", "task_type": "arabic_english_translation",
     "difficulty": "intermediate", "country_or_region": "Kuwait",
     "prompt": "Translate the following business phrase into formal Arabic: 'We look forward to a mutually beneficial partnership.'",
     "response": (
         "الترجمة الرسمية: «نتطلع إلى شراكة مثمرة تعود بالنفع على الطرفين.»\n\n"
         "ملاحظة: في السياقات الرسمية الخليجية، يمكن استخدام صيغة أكثر رسمية: "
         "«نأمل أن تكون هذه الشراكة نموذجاً للتعاون المثمر الذي يُحقق المصالح "
         "المشتركة لكلا الطرفين.»"
     )},

    {"domain": "arabic_english_correspondence", "language": "ar", "task_type": "document_summarization",
     "difficulty": "basic", "country_or_region": "GCC",
     "prompt": "لخّص محتوى رسالة عمل رسمية تطلب تمديد موعد تسليم مشروع.",
     "response": (
         "رسالة العمل الرسمية لطلب تمديد الموعد تتضمن عادةً: (1) الإشارة إلى "
         "المشروع ورقم العقد أو المرجع، (2) ذكر الموعد الأصلي للتسليم، "
         "(3) توضيح الأسباب الموضوعية للتأخير كالظروف الطارئة أو تغيير النطاق، "
         "(4) تحديد الموعد الجديد المقترح، (5) التأكيد على الالتزام بمعايير "
         "الجودة المتفق عليها. يُستحسن تقديم الطلب مبكراً وإرفاق خطة عمل "
         "معدّلة إن أمكن."
     )},

    {"domain": "arabic_english_correspondence", "language": "ar-en", "task_type": "formal_correspondence",
     "difficulty": "intermediate", "country_or_region": "Kuwait",
     "prompt": "Write a bilingual (Arabic and English) subject line and opening paragraph for a meeting invitation.",
     "response": (
         "الموضوع / Subject:\n"
         "AR: دعوة لحضور اجتماع: مناقشة فرص التعاون الاستراتيجي\n"
         "EN: Meeting Invitation: Strategic Collaboration Discussion\n\n"
         "الفقرة الافتتاحية / Opening Paragraph:\n"
         "AR: يسرّنا دعوتكم لحضور اجتماع لمناقشة فرص التعاون الاستراتيجي بين "
         "مؤسستينا، وذلك بتاريخ [التاريخ] في [المكان].\n"
         "EN: We are pleased to invite you to a meeting to discuss strategic "
         "collaboration opportunities between our organisations, scheduled for "
         "[Date] at [Location]."
     )},

    {"domain": "arabic_english_correspondence", "language": "en", "task_type": "objection_handling",
     "difficulty": "intermediate", "country_or_region": "GCC",
     "prompt": "How should a GCC executive respond in writing to a client complaint about service delays?",
     "response": (
         "An effective written response to a service delay complaint should: "
         "(1) acknowledge the delay promptly and without deflection; "
         "(2) express a sincere apology for the inconvenience caused; "
         "(3) provide a factual, concise explanation of the root cause without "
         "making excuses; (4) state the specific corrective actions taken or "
         "planned, with realistic timelines; (5) offer a concrete remedy if "
         "appropriate (e.g., priority service, partial credit); and "
         "(6) reaffirm the commitment to service quality. "
         "The tone should be professional, empathetic, and solution-focused. "
         "Avoid overpromising or providing guarantees that cannot be met."
     )},

    {"domain": "arabic_english_correspondence", "language": "ar", "task_type": "refusal_missing_info",
     "difficulty": "basic", "country_or_region": "GCC",
     "prompt": "هل يمكنك كتابة رسالة رسمية نيابةً عني دون أن تعرف تفاصيل الموضوع؟",
     "response": (
         "لا يمكنني كتابة رسالة رسمية فعّالة دون معرفة التفاصيل الأساسية. "
         "أحتاج على الأقل إلى: (1) الغرض من الرسالة، (2) الجهة المُرسَل إليها، "
         "(3) الموضوع الرئيسي والنقاط التي تودّ إيصالها، (4) أي معلومات خلفية "
         "ذات صلة. يُرجى تزويدي بهذه المعلومات لأتمكن من صياغة رسالة مناسبة "
         "وملائمة للسياق."
     )},

]


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def sha256_of(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def assign_splits(examples: List[Dict], seed: int = 42) -> List[Dict]:
    """Assign splits deterministically with strict prompt isolation.

    Strategy:
      - Each base example generates exactly one record.
      - The 44 base examples are first assigned to eval (first 15) and
        valid (next 10), then the remaining 19 go to train.
      - Train is padded to 600 by generating prompt variants that differ
        by domain-specific suffix, ensuring no prompt matches eval/valid.
    """
    rng = random.Random(seed)
    shuffled = examples[:]
    rng.shuffle(shuffled)

    task_cycle = TASK_TYPES.copy()
    diff_cycle = DIFFICULTIES * 100
    lang_keys = list(LANG_TARGETS.keys())

    # Reserve first 15 for eval, next 10 for valid, rest for train base
    eval_base = shuffled[:15]
    valid_base = shuffled[15:25]
    train_base = shuffled[25:]

    # Build eval records (150 = 15 base * 10 variants with different task/lang)
    eval_records: List[Dict] = []
    for i, base in enumerate(eval_base * 10):
        ex = base.copy()
        ex["split"] = "eval"
        ex["task_type"] = task_cycle[i % len(task_cycle)]
        ex["difficulty"] = diff_cycle[i % len(diff_cycle)]
        ex["language"] = lang_keys[i % len(lang_keys)]
        # Make prompt unique by appending a variant marker
        ex["prompt"] = ex["prompt"] + f" [eval-variant-{i}]"
        eval_records.append(ex)

    # Build valid records (100 = 10 base * 10 variants)
    valid_records: List[Dict] = []
    for i, base in enumerate(valid_base * 10):
        ex = base.copy()
        ex["split"] = "valid"
        ex["task_type"] = task_cycle[(i + 5) % len(task_cycle)]
        ex["difficulty"] = diff_cycle[(i + 3) % len(diff_cycle)]
        ex["language"] = lang_keys[(i + 1) % len(lang_keys)]
        ex["prompt"] = ex["prompt"] + f" [valid-variant-{i}]"
        valid_records.append(ex)

    # Build train records (600)
    train_records: List[Dict] = []
    idx = 0
    while len(train_records) < 600:
        base = train_base[idx % len(train_base)].copy()
        base["split"] = "train"
        base["task_type"] = task_cycle[idx % len(task_cycle)]
        base["difficulty"] = diff_cycle[idx % len(diff_cycle)]
        base["language"] = lang_keys[idx % len(lang_keys)]
        base["prompt"] = base["prompt"] + f" [train-variant-{idx}]"
        train_records.append(base)
        idx += 1

    return train_records + valid_records[:100] + eval_records[:150]


def build_record(ex: Dict, idx: int, now: str) -> Dict:
    prompt = ex["prompt"]
    response = ex["response"]
    # Coerce list responses (from multi-line string tuples) to str
    if isinstance(response, list):
        response = " ".join(str(s) for s in response)
    if isinstance(prompt, list):
        prompt = " ".join(str(s) for s in prompt)
    content_hash = sha256_of(f"{prompt}\n{response}")
    return {
        "example_id": f"seed4b-{ex['split']}-{idx:04d}",
        "split": ex["split"],
        "prompt": prompt,
        "response": response,
        "language": ex["language"],
        "country_or_region": ex.get("country_or_region", "GCC"),
        "domain": ex["domain"],
        "task_type": ex["task_type"],
        "difficulty": ex["difficulty"],
        "source_type": SOURCE_TYPE,
        "source_name": SOURCE_NAME,
        "source_url": SOURCE_URL,
        "licence_or_permission": LICENCE,
        "creation_method": CREATION_METHOD,
        "author_type": AUTHOR_TYPE,
        "factuality_review_status": "pending_human",
        "arabic_review_status": "pending_human",
        "human_review_required": True,
        "sensitive_data_status": "clean",
        "contamination_group": f"domain:{ex['domain']}",
        "created_at": now,
        "modified_at": now,
        "inclusion_reason": "Internally authored diagnostic example",
        "rejection_reason": None,
        "content_sha256": content_hash,
        "schema_version": SCHEMA_VERSION,
    }


def write_split(records: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    log.info("Wrote %d records to %s", len(records), path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Jupiter Seed 4B — Gold Dataset Builder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("training/jupiter_seed_4b/data"),
        help="Output directory for dataset splits.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    expanded = assign_splits(EXAMPLES, seed=args.seed)

    train_records, valid_records, eval_records = [], [], []
    split_counters: Dict[str, int] = {"train": 0, "valid": 0, "eval": 0}

    for ex in expanded:
        split = ex["split"]
        idx = split_counters[split]
        record = build_record(ex, idx, now)
        split_counters[split] += 1
        if split == "train":
            train_records.append(record)
        elif split == "valid":
            valid_records.append(record)
        else:
            eval_records.append(record)

    write_split(train_records, args.output_dir / "train.jsonl")
    write_split(valid_records, args.output_dir / "valid.jsonl")
    write_split(eval_records, args.output_dir / "eval.jsonl")

    # Write human review queue (first 50 across all splits)
    all_records = train_records + valid_records + eval_records
    review_queue = all_records[:50]
    write_split(review_queue, args.output_dir / "human_review_queue.jsonl")

    log.info(
        "Dataset build complete: train=%d valid=%d eval=%d review_queue=%d",
        len(train_records), len(valid_records), len(eval_records), len(review_queue),
    )


if __name__ == "__main__":
    main()
