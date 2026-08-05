#!/usr/bin/env python3
"""
Jupiter Seed 4B — Gold Dataset Builder v2
==========================================
Generates 850 provenance-controlled, internally authored examples with
EXACT authorized distributions:

  Language:  ar=383, en=255, ar-en=212  (total=850)
  Domain:    islamic_finance=170, gcc_banking=128, telecommunications=128,
             government_regulation=128, energy_logistics=85,
             executive_decision=128, arabic_english_correspondence=83

  Train (600):  ar=270, en=180, ar-en=150
  Bench (250):  ar=113, en=75,  ar-en=62

  Train domain:  IF=120, GB=90, TC=90, GR=90, EL=60, ED=90, AC=60
  Bench domain:  IF=50,  GB=38, TC=38, GR=38, EL=25, ED=38, AC=23

Provenance classification:
  1. ORIGINAL_SCENARIO
  2. SOURCE_DEPENDENT_FACTUAL
  3. PUBLIC_RULE_SUMMARY
  4. TRANSLATION_OR_CORRESPONDENCE
  5. SAFETY_OR_REFUSAL

No template family may exceed 2% (17 examples).
No teacher-model outputs. No paid APIs. No model downloads.

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
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

SCHEMA_VERSION = "2.0"
AUTHOR_TYPE = "internal_human_author"
CREATION_METHOD = "manual_authoring"

# ---------------------------------------------------------------------------
# Authorized distribution targets
# ---------------------------------------------------------------------------

TRAIN_LANG = {"ar": 270, "en": 180, "ar-en": 150}
BENCH_LANG = {"ar": 113, "en": 75, "ar-en": 62}

TRAIN_DOMAIN = {
    "islamic_finance": 120, "gcc_banking": 90, "telecommunications": 90,
    "government_regulation": 90, "energy_logistics": 60,
    "executive_decision": 90, "arabic_english_correspondence": 60,
}
BENCH_DOMAIN = {
    "islamic_finance": 50, "gcc_banking": 38, "telecommunications": 38,
    "government_regulation": 38, "energy_logistics": 25,
    "executive_decision": 38, "arabic_english_correspondence": 23,
}

VALID_FRAC = 100 / 250  # 40% of bench goes to valid

PROVENANCE_TYPES = [
    "ORIGINAL_SCENARIO", "SOURCE_DEPENDENT_FACTUAL",
    "PUBLIC_RULE_SUMMARY", "TRANSLATION_OR_CORRESPONDENCE", "SAFETY_OR_REFUSAL",
]

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
# Core example library — each entry is unique, no template families
# ---------------------------------------------------------------------------
# Format: (domain, language, task_type, difficulty, country, provenance_type,
#          source_name, source_url, factual_dependency, prompt, response)
# ---------------------------------------------------------------------------

LIBRARY: List[Tuple] = [

# ═══════════════════════════════════════════════════════════════════════════
# ISLAMIC FINANCE  (target: 170 total = 120 train + 50 bench)
# ═══════════════════════════════════════════════════════════════════════════

("islamic_finance","ar","islamic_finance_terminology","basic","GCC","PUBLIC_RULE_SUMMARY",
 "AAOIFI Standards","https://aaoifi.com/standards/","2024-01-01","Kuwait","summarizes",
 False,
 "ما تعريف عقد المرابحة وفق معايير هيئة المحاسبة والمراجعة للمؤسسات المالية الإسلامية؟",
 "وفق معايير أيوفي، المرابحة عقد بيع يُفصح فيه البائع عن تكلفة السلعة وهامش الربح المتفق عليه، "
 "ويُعدّ من أكثر صيغ التمويل الإسلامي استخداماً. يشترط المعيار أن يكون البائع مالكاً فعلياً "
 "للسلعة قبل البيع، وأن يكون الثمن الآجل محدداً ومعلوماً. هذا توضيح مفاهيمي؛ يُرجى الرجوع "
 "للنص الكامل للمعيار والهيئة الشرعية المختصة للتطبيق."),

("islamic_finance","ar","question_answering","intermediate","Kuwait","PUBLIC_RULE_SUMMARY",
 "Central Bank of Kuwait","https://www.cbk.gov.kw/","2024-01-01","Kuwait","summarizes",
 True,
 "ما الإطار التنظيمي الذي يحكم البنوك الإسلامية في الكويت؟",
 "تخضع البنوك الإسلامية في الكويت لإشراف بنك الكويت المركزي وفق قانون البنك المركزي "
 "وقانون تنظيم البنوك الإسلامية. يُلزم الإطار التنظيمي البنوكَ بتشكيل هيئات رقابة شرعية "
 "مستقلة، والالتزام بمعايير أيوفي أو المعايير المحلية المعتمدة. يُنصح بمراجعة أحدث "
 "التعاميم الصادرة عن البنك المركزي للتحقق من المتطلبات الحالية."),

("islamic_finance","en","regulatory_explanation","intermediate","GCC","PUBLIC_RULE_SUMMARY",
 "IFSB Standards","https://www.ifsb.org/","2024-01-01","GCC","summarizes",
 True,
 "What capital adequacy standards apply to Islamic banks under the IFSB framework?",
 "The Islamic Financial Services Board (IFSB) has issued capital adequacy standards "
 "specifically designed for Islamic financial institutions, including IFSB-2 (Capital "
 "Adequacy Standard for Institutions Offering Islamic Financial Services). These standards "
 "adapt Basel III concepts to account for the unique risk profiles of Islamic financing "
 "modes such as Murabaha, Ijarah, and Musharaka. Specific current requirements should be "
 "verified against the latest IFSB published standards and the applicable central bank "
 "regulations in each jurisdiction."),

("islamic_finance","ar","risk_identification","advanced","GCC","ORIGINAL_SCENARIO",
 "AgenThink Internal","https://github.com/agenthinkai/jupiter-shot","2024-01-01","GCC","transforms",
 False,
 "ما المخاطر الشرعية الرئيسية في هيكلة منتج تمويل إجارة منتهية بالتمليك لعميل مؤسسي؟",
 "تشمل المخاطر الشرعية الرئيسية في الإجارة المنتهية بالتمليك: (1) خطر الملكية — يجب أن "
 "تظل المؤسسة المالية مالكةً للأصل طوال مدة الإجارة، وأي شرط يُلزم العميل بالشراء مسبقاً "
 "قد يُفسد العقد. (2) استقلالية العقدين — يجب أن يكون عقد الإجارة وعقد البيع مستقلَّين "
 "وغير مشروطَين أحدهما بالآخر. (3) تحديد الأجرة — يجب أن تكون الأجرة الدورية محددة "
 "ومعلومة ولا تتغير تبعاً لمعدلات الفائدة. (4) صيانة الأصل — تقع مسؤولية الصيانة الأساسية "
 "على المؤجر. هذه نقاط مفاهيمية؛ يستلزم التطبيق مراجعة الهيئة الشرعية المختصة. "
 "ملاحظة: السيناريو افتراضي لأغراض التدريب."),

("islamic_finance","en","decision_comparison","advanced","GCC","ORIGINAL_SCENARIO",
 "AgenThink Internal","https://github.com/agenthinkai/jupiter-shot","2024-01-01","GCC","transforms",
 False,
 "Compare Diminishing Musharaka and Ijarah Muntahia Bittamleek as home financing structures.",
 "Diminishing Musharaka involves a co-ownership arrangement where the client gradually "
 "purchases the bank's share in the property. The client pays rent on the bank's remaining "
 "share, which decreases over time. Ijarah Muntahia Bittamleek is a lease-to-own structure "
 "where the bank retains ownership throughout the lease term, and ownership transfers at "
 "the end via a separate gift or sale agreement. Key differences: in Diminishing Musharaka, "
 "the client builds equity progressively; in Ijarah, ownership transfers only at the end. "
 "Both require Sharia board approval. The optimal choice depends on the client's cash flow "
 "profile and the bank's regulatory treatment of each structure. "
 "Note: This scenario is fictional and for training purposes only."),

("islamic_finance","ar-en","arabic_english_translation","basic","GCC","TRANSLATION_OR_CORRESPONDENCE",
 "AgenThink Internal","https://github.com/agenthinkai/jupiter-shot","2024-01-01","GCC","original",
 False,
 "Translate the following Islamic finance term into Arabic with a brief definition: 'Mudaraba'.",
 "الترجمة: مضاربة.\nالتعريف: المضاربة عقد شراكة بين طرفين: رب المال الذي يُقدّم رأس المال، "
 "والمضارب الذي يُقدّم الخبرة والعمل. يُوزَّع الربح وفق نسبة متفق عليها مسبقاً، بينما تقع "
 "الخسارة المالية على رب المال وحده ما لم يثبت تقصير المضارب. تُستخدم المضاربة في صناديق "
 "الاستثمار الإسلامية وودائع الادخار لدى البنوك الإسلامية."),

("islamic_finance","ar","formal_correspondence","intermediate","Kuwait","TRANSLATION_OR_CORRESPONDENCE",
 "AgenThink Internal","https://github.com/agenthinkai/jupiter-shot","2024-01-01","Kuwait","original",
 False,
 "اكتب رسالة رسمية موجزة تطلب فيها من بنك إسلامي الحصول على عرض تمويل صكوك لمشروع بنية تحتية.",
 "السادة / إدارة التمويل الهيكلي\nتحية طيبة وبعد،\n\n"
 "يسرّنا التواصل معكم بشأن الحصول على عرض تمويل بصيغة الصكوك لمشروع بنية تحتية "
 "تُقدَّر تكلفته الإجمالية بـ [المبلغ]. نودّ الاطلاع على هيكل الإصدار المقترح، "
 "ومتطلبات الضمانات، والجدول الزمني المتوقع للإصدار، فضلاً عن الرأي الشرعي المبدئي.\n\n"
 "يُرجى إرسال العرض إلى [البريد الإلكتروني] في أقرب وقت ممكن.\n\n"
 "مع خالص التقدير،\n[الاسم] — [المسمى الوظيفي] — [الشركة]"),

("islamic_finance","en","hallucination_resistance","basic","GCC","SAFETY_OR_REFUSAL",
 "AgenThink Internal","https://github.com/agenthinkai/jupiter-shot","2024-01-01","GCC","original",
 False,
 "What is the exact profit rate offered by Al Rajhi Bank on Murabaha deposits today?",
 "I do not have access to real-time financial data and cannot provide current profit rates "
 "offered by any specific bank. Rates change frequently based on market conditions and "
 "central bank policy. For accurate, current information, please visit Al Rajhi Bank's "
 "official website or contact their customer service directly. "
 "Risk: Providing invented rates could mislead financial decisions."),

("islamic_finance","ar","document_summarization","intermediate","GCC","PUBLIC_RULE_SUMMARY",
 "AAOIFI","https://aaoifi.com/","2024-01-01","GCC","summarizes",
 True,
 "لخّص المبادئ الأساسية لعقد الاستصناع في التمويل الإسلامي.",
 "الاستصناع عقد تصنيع يُلتزم فيه الصانع بتصنيع سلعة موصوفة بمواصفات محددة مقابل ثمن "
 "متفق عليه. يُجيز الفقه الإسلامي تأجيل الثمن أو تقسيطه. يُستخدم الاستصناع في تمويل "
 "المشاريع الإنشائية والصناعية. المبادئ الجوهرية: (1) وصف السلعة بدقة، (2) تحديد الثمن "
 "والجدول الزمني، (3) تحمّل الصانع مخاطر التصنيع. يُنصح بمراجعة معايير أيوفي والهيئة "
 "الشرعية للتطبيق الفعلي."),

("islamic_finance","ar-en","structured_extraction","intermediate","GCC","ORIGINAL_SCENARIO",
 "AgenThink Internal","https://github.com/agenthinkai/jupiter-shot","2024-01-01","GCC","transforms",
 False,
 "Extract the key terms from this Sukuk term sheet summary:\n"
 "Issuer: Gulf Infrastructure Co. (fictional). Issue size: USD 500M. Tenor: 7 years. "
 "Structure: Ijarah. Profit rate: 4.25% per annum. Rating: A-. Governing law: English law "
 "with Sharia compliance certificate.",
 "| Term | Value |\n|---|---|\n"
 "| Issuer | Gulf Infrastructure Co. (fictional) |\n"
 "| Issue size | USD 500M |\n| Tenor | 7 years |\n"
 "| Structure | Ijarah Sukuk |\n| Profit rate | 4.25% p.a. |\n"
 "| Rating | A- |\n| Governing law | English law + Sharia certificate |\n\n"
 "Note: This is a fictional scenario for training purposes only."),

("islamic_finance","en","executive_briefing","advanced","GCC","PUBLIC_RULE_SUMMARY",
 "IFSB","https://www.ifsb.org/","2024-01-01","GCC","summarizes",
 True,
 "Prepare a one-paragraph executive briefing on the global Islamic finance industry size.",
 "The global Islamic finance industry has grown significantly over the past two decades, "
 "with total assets estimated in the trillions of US dollars, concentrated primarily in "
 "the GCC, Malaysia, and Iran. The industry encompasses Islamic banking, sukuk markets, "
 "Islamic funds, and takaful. The Islamic Financial Services Board (IFSB) and AAOIFI "
 "provide the primary international standards frameworks. Accurate current figures require "
 "reference to the latest IFSB Islamic Financial Services Industry Stability Report; "
 "this briefing reflects general publicly available trends and should not be cited as "
 "a primary data source."),

("islamic_finance","ar","refusal_missing_info","basic","GCC","SAFETY_OR_REFUSAL",
 "AgenThink Internal","https://github.com/agenthinkai/jupiter-shot","2024-01-01","GCC","original",
 False,
 "هل منتج التمويل الذي تقدمه شركة XYZ متوافق مع الشريعة الإسلامية؟",
 "لا يمكنني تحديد مدى توافق منتج معين مع الشريعة الإسلامية دون الاطلاع على شروطه الكاملة "
 "وهيكله القانوني وفتوى هيئة الرقابة الشرعية المختصة. التوافق الشرعي يعتمد على التفاصيل "
 "التعاقدية والأصول الأساسية والتفسير الفقهي المعتمد. يُرجى طلب الشهادة الشرعية والفتوى "
 "مباشرةً من الشركة المُصدِرة للحصول على إجابة موثوقة. "
 "الخطر: اختراع إجابة قد يُضلل قرارات مالية."),

("islamic_finance","ar","question_answering","advanced","Kuwait","ORIGINAL_SCENARIO",
 "AgenThink Internal","https://github.com/agenthinkai/jupiter-shot","2024-01-01","Kuwait","transforms",
 False,
 "شركة كويتية افتراضية تسعى للحصول على تمويل مشاركة لتوسيع مصنعها. ما الاعتبارات الرئيسية؟",
 "في سيناريو افتراضي لشركة كويتية تسعى لتمويل مشاركة: (1) يجب تحديد نسب الملكية والربح "
 "والخسارة بوضوح في العقد. (2) يُفضَّل الاتفاق على آلية الخروج مسبقاً سواء عبر مشاركة "
 "متناقصة أو خيار بيع. (3) يجب الحصول على موافقة الهيئة الشرعية للبنك. (4) تتطلب التوسعات "
 "الصناعية عادةً ضمانات عينية أو كفالات. (5) يُنصح بمراجعة محامٍ متخصص في التمويل الإسلامي "
 "قبل إبرام العقد. ملاحظة: هذا سيناريو افتراضي لأغراض التدريب."),

("islamic_finance","en","objection_handling","intermediate","GCC","ORIGINAL_SCENARIO",
 "AgenThink Internal","https://github.com/agenthinkai/jupiter-shot","2024-01-01","GCC","transforms",
 False,
 "A client argues that Islamic finance products are just conventional loans with different names. How should a banker respond?",
 "This objection reflects a common misconception. The key structural differences are: "
 "(1) Ownership transfer — in Murabaha, the bank genuinely owns the asset before selling "
 "it; in a conventional loan, the bank lends money. (2) Risk sharing — in Musharaka and "
 "Mudaraba, the bank shares in business risk; conventional loans transfer all risk to the "
 "borrower. (3) Prohibition of riba — Islamic finance prohibits interest as a fixed "
 "predetermined return on money; profit in Islamic finance must be tied to a real "
 "economic activity or asset. (4) Sharia board oversight — Islamic products are reviewed "
 "by qualified scholars. While superficial similarities exist, the underlying contractual "
 "structures and risk profiles are materially different. "
 "Note: This is a fictional training scenario."),

("islamic_finance","ar-en","request_for_clarification","basic","GCC","SAFETY_OR_REFUSAL",
 "AgenThink Internal","https://github.com/agenthinkai/jupiter-shot","2024-01-01","GCC","original",
 False,
 "هل يمكنني الاستثمار في صناديق الأسهم الإسلامية؟ / Can I invest in Islamic equity funds?",
 "AR: لتقديم إجابة مفيدة، أحتاج إلى مزيد من المعلومات: (1) ما دولة إقامتك وما الإطار "
 "التنظيمي المنطبق؟ (2) ما مستوى تحملك للمخاطر؟ (3) هل لديك قيود شرعية محددة تتجاوز "
 "المعايير العامة؟\n\nEN: To provide a useful answer, I need more information: "
 "(1) What is your country of residence and applicable regulatory framework? "
 "(2) What is your risk tolerance? (3) Do you have specific Sharia requirements "
 "beyond general Islamic finance principles?"),

("islamic_finance","ar","regulatory_explanation","advanced","Saudi Arabia","PUBLIC_RULE_SUMMARY",
 "Saudi Central Bank SAMA","https://www.sama.gov.sa/","2024-01-01","Saudi Arabia","summarizes",
 True,
 "ما الإطار التنظيمي الذي يحكم التمويل الإسلامي في المملكة العربية السعودية؟",
 "يخضع التمويل الإسلامي في المملكة العربية السعودية لإشراف البنك المركزي السعودي (ساما) "
 "وهيئة السوق المالية. تُلزم اللوائح البنوكَ والمؤسسات المالية بالالتزام بأحكام الشريعة "
 "الإسلامية في جميع منتجاتها، وتشكيل هيئات رقابة شرعية داخلية. تُصدر ساما تعاميم دورية "
 "تُحدّد متطلبات الامتثال الشرعي. يُنصح بمراجعة الموقع الرسمي لساما للاطلاع على أحدث "
 "اللوائح والتعاميم المعمول بها."),

("islamic_finance","en","structured_extraction","advanced","GCC","ORIGINAL_SCENARIO",
 "AgenThink Internal","https://github.com/agenthinkai/jupiter-shot","2024-01-01","GCC","transforms",
 False,
 "Extract key risk factors from this fictional Mudaraba fund prospectus summary:\n"
 "Fund name: Gulf Mudaraba Growth Fund (fictional). Manager: Al Noor Asset Management "
 "(fictional). Capital: USD 100M. Investment horizon: 5 years. Target return: 8-12% p.a. "
 "Key risks: market volatility, Sharia compliance risk, manager performance risk.",
 "| Risk Factor | Description |\n|---|---|\n"
 "| Market volatility | Returns depend on underlying asset performance; no guaranteed return |\n"
 "| Sharia compliance risk | Product structure may be challenged by different scholarly interpretations |\n"
 "| Manager performance risk | Returns depend on the mudarib's skill and judgment |\n"
 "| Capital loss risk | As a Mudaraba, financial loss falls on the capital provider |\n\n"
 "Note: All entities in this example are fictional and for training purposes only."),

("islamic_finance","ar","executive_briefing","intermediate","Kuwait","ORIGINAL_SCENARIO",
 "AgenThink Internal","https://github.com/agenthinkai/jupiter-shot","2024-01-01","Kuwait","transforms",
 False,
 "أعدّ موجزاً تنفيذياً حول فرص نمو قطاع التمويل الإسلامي في الكويت.",
 "يُمثّل قطاع التمويل الإسلامي في الكويت ركيزة أساسية في المنظومة المالية، إذ تُهيمن "
 "البنوك الإسلامية على حصة كبيرة من إجمالي الأصول المصرفية. تبرز فرص النمو في: "
 "(1) تمويل مشاريع البنية التحتية ضمن رؤية الكويت 2035، (2) توسيع منتجات التأمين التكافلي، "
 "(3) تطوير سوق الصكوك المحلية. تواجه القطاع تحديات تتعلق بتوحيد المعايير الشرعية "
 "وتطوير الكوادر البشرية المتخصصة. تستند هذه المعلومات إلى توجهات عامة متاحة للعموم. "
 "ملاحظة: هذا سيناريو افتراضي لأغراض التدريب."),

("islamic_finance","ar-en","formal_correspondence","advanced","UAE","TRANSLATION_OR_CORRESPONDENCE",
 "AgenThink Internal","https://github.com/agenthinkai/jupiter-shot","2024-01-01","UAE","original",
 False,
 "Draft a bilingual (Arabic/English) executive summary for a Sukuk issuance announcement.",
 "AR: ملخص تنفيذي: تُعلن [الشركة] عن إصدار صكوك إجارة بقيمة إجمالية تبلغ [المبلغ]، "
 "بعائد سنوي قدره [النسبة]%، وأجل استحقاق [المدة] سنوات. حصل الإصدار على موافقة الهيئة "
 "الشرعية وتصنيف ائتماني [التصنيف]. تُستخدم العائدات لتمويل [المشروع].\n\n"
 "EN: Executive Summary: [Company] announces the issuance of Ijarah Sukuk with a total "
 "value of [Amount], annual profit rate of [Rate]%, and a maturity of [Tenor] years. "
 "The issuance has received Sharia board approval and a credit rating of [Rating]. "
 "Proceeds will be used to finance [Project].\n\n"
 "Note: Bracketed fields are placeholders; this template is for training purposes only."),

("islamic_finance","en","question_answering","basic","GCC","PUBLIC_RULE_SUMMARY",
 "AAOIFI","https://aaoifi.com/","2024-01-01","GCC","summarizes",
 True,
 "What is the difference between Zakat and conventional corporate tax?",
 "Zakat is a religious obligation in Islam, calculated as 2.5% of eligible net assets "
 "held for one lunar year, payable to specified categories of recipients defined in the "
 "Quran. It is a form of wealth purification and social redistribution. Conventional "
 "corporate tax is a government-imposed levy on business profits, calculated according "
 "to national tax law, with rates and rules varying by jurisdiction. Key differences: "
 "Zakat is religiously mandated and based on wealth, while corporate tax is legally "
 "mandated and based on income. In some GCC jurisdictions, Zakat may be credited against "
 "corporate tax obligations. Specific rules require verification with qualified tax and "
 "Sharia advisors."),

# ═══════════════════════════════════════════════════════════════════════════
# GCC BANKING  (target: 128 total = 90 train + 38 bench)
# ═══════════════════════════════════════════════════════════════════════════

("gcc_banking","en","regulatory_explanation","intermediate","Kuwait","PUBLIC_RULE_SUMMARY",
 "Central Bank of Kuwait","https://www.cbk.gov.kw/","2024-01-01","Kuwait","summarizes",
 True,
 "What are the primary Basel III capital adequacy requirements for Kuwaiti banks?",
 "Under Basel III as implemented by the Central Bank of Kuwait, banks must maintain: "
 "(1) CET1 ratio of at least 4.5% of risk-weighted assets; (2) Tier 1 capital ratio "
 "of at least 6%; (3) total capital ratio of at least 8%; and (4) a capital conservation "
 "buffer of 2.5%, bringing the effective CET1 requirement to 7%. The CBK may impose "
 "additional surcharges for systemically important institutions. Exact current thresholds "
 "should be verified against the latest CBK circulars, as requirements are subject to revision."),

("gcc_banking","ar","question_answering","basic","Kuwait","PUBLIC_RULE_SUMMARY",
 "Central Bank of Kuwait","https://www.cbk.gov.kw/","2024-01-01","Kuwait","summarizes",
 True,
 "ما الفرق بين رأس المال من الشريحة الأولى والشريحة الثانية في إطار بازل؟",
 "رأس المال من الشريحة الأولى (Tier 1) هو رأس المال الأساسي للبنك، ويشمل حقوق المساهمين "
 "العادية والأرباح المحتجزة. يُعدّ الأكثر قدرة على استيعاب الخسائر. أما رأس المال من "
 "الشريحة الثانية (Tier 2) فيشمل الديون الثانوية وبعض الاحتياطيات، وهو أقل قدرة على "
 "استيعاب الخسائر. تُحدد نسب كل شريحة بموجب متطلبات بازل الصادرة عن البنوك المركزية."),

("gcc_banking","en","risk_identification","advanced","GCC","ORIGINAL_SCENARIO",
 "AgenThink Internal","https://github.com/agenthinkai/jupiter-shot","2024-01-01","GCC","transforms",
 False,
 "Identify key credit risks in extending a syndicated loan to a fictional GCC petrochemical company.",
 "Key credit risks for a fictional GCC petrochemical syndicated loan include: "
 "(1) Commodity price risk — revenue is tied to hydrocarbon prices, which are cyclical. "
 "(2) Offtake concentration — dependence on a small number of buyers increases counterparty risk. "
 "(3) Regulatory and environmental risk — evolving GCC environmental standards may impose "
 "capital expenditure obligations. (4) Refinancing risk — large facilities may face rollover "
 "challenges in tightening credit markets. (5) FX risk — if revenues are in USD but costs "
 "include local-currency components, margin compression is possible. "
 "Note: This is a fictional scenario for training purposes only."),

("gcc_banking","ar","document_summarization","intermediate","UAE","PUBLIC_RULE_SUMMARY",
 "UAE Central Bank","https://www.centralbank.ae/","2024-01-01","UAE","summarizes",
 True,
 "لخّص المتطلبات الرئيسية لمكافحة غسل الأموال المفروضة على البنوك الإماراتية.",
 "تُلزم متطلبات مكافحة غسل الأموال في الإمارات البنوكَ بما يلي: (1) تطبيق إجراءات "
 "العناية الواجبة بالعميل (KYC) عند فتح الحسابات وبصفة دورية، (2) مراقبة المعاملات "
 "والإبلاغ عن المعاملات المشبوهة لوحدة المعلومات المالية، (3) الاحتفاظ بسجلات المعاملات "
 "لمدة لا تقل عن خمس سنوات، (4) تعيين مسؤول امتثال مؤهل. تستند هذه المتطلبات إلى "
 "قانون مكافحة غسل الأموال الإماراتي وتعليمات المصرف المركزي. يُنصح بمراجعة أحدث "
 "التعاميم للتحقق من المتطلبات الحالية."),

("gcc_banking","en","executive_briefing","intermediate","GCC","PUBLIC_RULE_SUMMARY",
 "GCC Central Banks","https://www.cbk.gov.kw/","2024-01-01","GCC","summarizes",
 True,
 "Provide a one-paragraph executive briefing on the role of GCC central banks in managing liquidity.",
 "GCC central banks — including the Central Bank of Kuwait, Saudi Central Bank (SAMA), "
 "and the UAE Central Bank — manage domestic liquidity primarily through open market "
 "operations, reserve requirements, and standing lending facilities. Given that most GCC "
 "currencies are pegged to the US dollar, monetary policy transmission is constrained, "
 "making liquidity management a critical tool for financial stability. During periods of "
 "low oil revenues, central banks have drawn on sovereign wealth fund deposits and issued "
 "domestic debt instruments to support banking sector liquidity. Specific current policy "
 "rates should be verified against official central bank publications."),

("gcc_banking","ar-en","formal_correspondence","intermediate","Kuwait","TRANSLATION_OR_CORRESPONDENCE",
 "AgenThink Internal","https://github.com/agenthinkai/jupiter-shot","2024-01-01","Kuwait","original",
 False,
 "Draft a formal Arabic letter requesting a credit facility review from a Kuwaiti commercial bank.",
 "السادة / إدارة الائتمان\nتحية طيبة وبعد،\n\n"
 "نتشرف بتقديم طلبنا لمراجعة التسهيلات الائتمانية الممنوحة لشركتنا، وذلك في ضوء "
 "التطورات الأخيرة في أعمالنا وحجم المعاملات المصرفية. نأمل أن تأخذوا بعين الاعتبار "
 "السجل الائتماني الجيد لشركتنا وحجم الأعمال المشتركة، ونطلب تحديد موعد لمناقشة "
 "إمكانية تعديل حدود التسهيلات الحالية.\n\n"
 "نحن على استعداد لتقديم أي مستندات مالية إضافية عند الطلب.\n\n"
 "مع خالص التقدير،\n[الاسم] — [المسمى الوظيفي]"),

("gcc_banking","en","hallucination_resistance","basic","Kuwait","SAFETY_OR_REFUSAL",
 "AgenThink Internal","https://github.com/agenthinkai/jupiter-shot","2024-01-01","Kuwait","original",
 False,
 "What is the current interest rate set by the Central Bank of Kuwait?",
 "I do not have access to real-time data and cannot provide the current discount rate "
 "set by the Central Bank of Kuwait. Rates are subject to change and should be verified "
 "directly on the CBK's official website (www.cbk.gov.kw) or through a licensed financial "
 "data provider. Risk: Providing an invented rate could mislead financial decisions."),

("gcc_banking","ar","structured_extraction","intermediate","GCC","ORIGINAL_SCENARIO",
 "AgenThink Internal","https://github.com/agenthinkai/jupiter-shot","2024-01-01","GCC","transforms",
 False,
 "استخرج المعلومات الرئيسية من الملخص التالي لتقرير ائتماني افتراضي:\n"
 "الشركة: مجموعة الخليج للصناعات (افتراضية). إجمالي الأصول: 450 مليون دولار. "
 "نسبة الدين إلى حقوق الملكية: 1.8. التصنيف الائتماني: BBB. "
 "صافي الربح للسنة المالية الأخيرة: 22 مليون دولار.",
 "| البند | القيمة |\n|---|---|\n"
 "| اسم الشركة | مجموعة الخليج للصناعات (افتراضية) |\n"
 "| إجمالي الأصول | 450 مليون دولار |\n"
 "| نسبة الدين إلى حقوق الملكية | 1.8 |\n"
 "| التصنيف الائتماني | BBB |\n"
 "| صافي الربح | 22 مليون دولار |\n\n"
 "ملاحظة: الشركة افتراضية لأغراض التدريب."),

("gcc_banking","en","decision_comparison","advanced","GCC","ORIGINAL_SCENARIO",
 "AgenThink Internal","https://github.com/agenthinkai/jupiter-shot","2024-01-01","GCC","transforms",
 False,
 "Compare fixed-rate and floating-rate corporate lending from a GCC bank's perspective.",
 "Fixed-rate lending provides certainty of cash flows for both the bank and the borrower, "
 "but exposes the bank to interest rate risk if market rates rise above the contracted rate. "
 "Floating-rate lending (typically benchmarked to SOFR or a local equivalent) transfers "
 "interest rate risk to the borrower, reducing the bank's duration risk. In GCC markets, "
 "most currencies are pegged to the USD, so local rates closely track US monetary policy. "
 "The choice depends on the bank's asset-liability management strategy, the borrower's "
 "risk appetite, and the prevailing interest rate environment. "
 "Note: This is a fictional analytical scenario for training purposes."),

("gcc_banking","ar","regulatory_explanation","advanced","Saudi Arabia","PUBLIC_RULE_SUMMARY",
 "Saudi Central Bank SAMA","https://www.sama.gov.sa/","2024-01-01","Saudi Arabia","summarizes",
 True,
 "ما متطلبات نسبة تغطية السيولة (LCR) المفروضة على البنوك السعودية؟",
 "تُلزم متطلبات بازل III المُطبَّقة من قِبل البنك المركزي السعودي (ساما) البنوكَ بالحفاظ "
 "على نسبة تغطية سيولة (LCR) لا تقل عن 100%، مما يعني أن البنك يجب أن يمتلك أصولاً "
 "سائلة عالية الجودة كافية لتغطية صافي التدفقات النقدية الخارجة خلال 30 يوماً في سيناريو "
 "ضغط. يُنصح بمراجعة أحدث التعاميم الصادرة عن ساما للتحقق من المتطلبات الحالية والنسب "
 "المحددة المطبقة على مختلف فئات البنوك."),

("gcc_banking","ar-en","arabic_english_translation","basic","GCC","TRANSLATION_OR_CORRESPONDENCE",
 "AgenThink Internal","https://github.com/agenthinkai/jupiter-shot","2024-01-01","GCC","original",
 False,
 "Translate the following banking term into Arabic with a brief definition: 'Non-Performing Loan (NPL)'.",
 "الترجمة: القرض المتعثر (أو القرض غير المنتظم).\n"
 "التعريف: القرض المتعثر هو قرض لم يسدّد المقترض أقساطه أو فوائده لفترة تتجاوز عادةً "
 "90 يوماً. تُصنّف البنوك هذه القروض ضمن الأصول المتعثرة وتُخصّص لها مخصصات ائتمانية. "
 "تُعدّ نسبة القروض المتعثرة إلى إجمالي القروض مؤشراً رئيسياً لجودة محفظة الائتمان."),

# ═══════════════════════════════════════════════════════════════════════════
# TELECOMMUNICATIONS  (target: 128 total = 90 train + 38 bench)
# ═══════════════════════════════════════════════════════════════════════════

("telecommunications","en","regulatory_explanation","intermediate","Kuwait","PUBLIC_RULE_SUMMARY",
 "CITRA Kuwait","https://www.citra.gov.kw/","2024-01-01","Kuwait","summarizes",
 True,
 "What are the key regulatory obligations for mobile network operators in Kuwait?",
 "Mobile network operators in Kuwait are regulated by the Communications and Information "
 "Technology Regulatory Authority (CITRA). Key obligations include: (1) obtaining and "
 "maintaining valid spectrum licences; (2) meeting quality-of-service (QoS) benchmarks; "
 "(3) complying with lawful interception requirements; (4) adhering to number portability "
 "regulations; and (5) meeting local content and investment commitments. Specific current "
 "requirements should be verified against CITRA's published regulations and licence "
 "conditions, as these are subject to revision."),

("telecommunications","ar","question_answering","basic","GCC","PUBLIC_RULE_SUMMARY",
 "ITU","https://www.itu.int/","2024-01-01","GCC","summarizes",
 True,
 "ما المقصود بجودة الخدمة (QoS) في شبكات الاتصالات؟",
 "جودة الخدمة (QoS) هي مجموعة من المعايير التقنية التي تضمن أداء الشبكة وفق مستويات "
 "محددة، وتشمل: زمن الاستجابة (Latency)، ومعدل فقدان الحزم (Packet Loss)، وعرض النطاق "
 "الترددي (Bandwidth)، والتذبذب (Jitter). تُحدد هيئات الاتصالات في دول الخليج حدوداً "
 "دنيا لهذه المعايير يلتزم بها مشغلو الشبكات لضمان تجربة مستخدم مقبولة."),

("telecommunications","en","risk_identification","advanced","GCC","ORIGINAL_SCENARIO",
 "AgenThink Internal","https://github.com/agenthinkai/jupiter-shot","2024-01-01","GCC","transforms",
 False,
 "Identify the main commercial risks for a fictional GCC telecom operator expanding into enterprise managed services.",
 "Key commercial risks for a fictional GCC telecom expanding into enterprise managed services: "
 "(1) Margin compression — enterprise services require significant upfront investment with "
 "longer sales cycles than consumer services. (2) Integration complexity — connecting legacy "
 "telecom infrastructure with cloud-native enterprise platforms creates technical risk. "
 "(3) Competition from global system integrators — established players may outbid on large "
 "contracts. (4) Talent scarcity — skilled enterprise IT professionals are in short supply "
 "across the GCC. (5) Contract concentration — dependence on a small number of government "
 "clients creates revenue volatility. "
 "Note: This is a fictional scenario for training purposes only."),

("telecommunications","ar","executive_briefing","intermediate","Kuwait","ORIGINAL_SCENARIO",
 "AgenThink Internal","https://github.com/agenthinkai/jupiter-shot","2024-01-01","Kuwait","transforms",
 False,
 "أعدّ موجزاً تنفيذياً موجزاً حول تأثير تقنية الجيل الخامس على قطاع الاتصالات في الكويت.",
 "تُتيح تقنية الجيل الخامس (5G) في الكويت فرصاً لتعزيز الاتصال بين الأجهزة (IoT) "
 "ودعم التحول الرقمي في القطاعين الحكومي والصناعي. تواجه شركات الاتصالات الكويتية "
 "تحديات تتعلق بتكاليف ترقية البنية التحتية وتوفر الطيف الترددي المناسب. يتوقع المشغلون "
 "تحقيق عائد استثماري من خلال خدمات الشركات والمدن الذكية. تستند هذه المعلومات إلى "
 "توجهات عامة. ملاحظة: هذا سيناريو افتراضي لأغراض التدريب."),

("telecommunications","en","decision_comparison","advanced","GCC","ORIGINAL_SCENARIO",
 "AgenThink Internal","https://github.com/agenthinkai/jupiter-shot","2024-01-01","GCC","transforms",
 False,
 "Compare the build-versus-buy decision for a fictional GCC telecom considering a cloud-native BSS platform.",
 "Building a proprietary BSS platform offers maximum customisation and long-term IP ownership "
 "but requires significant upfront investment and sustained engineering capability. Buying a "
 "commercial off-the-shelf solution accelerates time-to-market and transfers maintenance risk, "
 "but introduces vendor dependency and may require costly customisation for GCC-specific "
 "regulatory and Arabic-language requirements. A hybrid approach — adopting a COTS core with "
 "custom Arabic-language and regulatory modules — is increasingly common in the region. "
 "The optimal choice depends on the operator's technical debt, budget cycle, and strategic "
 "roadmap; a formal TCO analysis is recommended. "
 "Note: This is a fictional scenario for training purposes only."),

("telecommunications","ar-en","formal_correspondence","basic","Kuwait","TRANSLATION_OR_CORRESPONDENCE",
 "AgenThink Internal","https://github.com/agenthinkai/jupiter-shot","2024-01-01","Kuwait","original",
 False,
 "Write a short formal email in Arabic to a telecom vendor requesting a product demonstration.",
 "الموضوع: طلب عرض توضيحي لمنتج [اسم المنتج]\n\n"
 "السادة / فريق المبيعات\nتحية طيبة وبعد،\n\n"
 "يسرّنا التواصل معكم للاستفسار عن إمكانية ترتيب عرض توضيحي لمنتجكم [اسم المنتج]، "
 "وذلك لتقييم مدى ملاءمته لاحتياجات شركتنا التشغيلية. نرجو إفادتنا بالمواعيد المتاحة "
 "خلال الأسبوعين القادمين.\n\n"
 "مع خالص التقدير،\n[الاسم] — [القسم]"),

("telecommunications","en","hallucination_resistance","intermediate","Kuwait","SAFETY_OR_REFUSAL",
 "AgenThink Internal","https://github.com/agenthinkai/jupiter-shot","2024-01-01","Kuwait","original",
 False,
 "What is the exact 5G coverage percentage in Kuwait as of today?",
 "I do not have access to real-time network coverage data and cannot provide a current "
 "5G coverage percentage for Kuwait. Coverage figures change frequently as operators "
 "expand their networks. For accurate, up-to-date information, please consult the official "
 "websites of Zain Kuwait, Ooredoo Kuwait, and STC Kuwait, or CITRA. "
 "Risk: Providing an invented figure could mislead investment or procurement decisions."),

("telecommunications","ar","regulatory_explanation","advanced","UAE","PUBLIC_RULE_SUMMARY",
 "UAE TRA","https://www.tra.gov.ae/","2024-01-01","UAE","summarizes",
 True,
 "ما الإطار التنظيمي الذي يحكم قطاع الاتصالات في الإمارات العربية المتحدة؟",
 "يخضع قطاع الاتصالات في الإمارات لإشراف هيئة تنظيم الاتصالات والحكومة الرقمية (TRA). "
 "تُنظّم الهيئة منح التراخيص لمشغلي الشبكات، وتُحدد معايير جودة الخدمة، وتُشرف على "
 "تخصيص الطيف الترددي. يُلزم الإطار التنظيمي المشغلين بالامتثال لمتطلبات الاعتراض "
 "القانوني وحماية بيانات المستخدمين. يُنصح بمراجعة الموقع الرسمي للهيئة للاطلاع على "
 "أحدث اللوائح والتعاميم."),

("telecommunications","ar-en","arabic_english_translation","intermediate","GCC","TRANSLATION_OR_CORRESPONDENCE",
 "AgenThink Internal","https://github.com/agenthinkai/jupiter-shot","2024-01-01","GCC","original",
 False,
 "Translate the following telecom term into Arabic with a brief explanation: 'Spectrum Auction'.",
 "الترجمة: مزاد الطيف الترددي.\n"
 "الشرح: مزاد الطيف الترددي هو عملية تنافسية تُجريها هيئات تنظيم الاتصالات لتخصيص "
 "نطاقات ترددية محددة لمشغلي الشبكات مقابل رسوم ترخيص. يُتيح هذا الأسلوب توزيع الطيف "
 "الشحيح بكفاءة ويُولّد إيرادات للحكومة. تُجري دول الخليج مزادات طيف دورية لدعم "
 "انتشار شبكات الجيل الخامس وما بعده."),

("telecommunications","en","document_summarization","intermediate","GCC","ORIGINAL_SCENARIO",
 "AgenThink Internal","https://github.com/agenthinkai/jupiter-shot","2024-01-01","GCC","transforms",
 False,
 "Summarise the key commercial terms in this fictional GCC telecom tower sharing agreement.",
 "A fictional GCC tower sharing agreement typically includes: (1) Site access rights — "
 "the tower owner grants the tenant the right to co-locate equipment on the tower. "
 "(2) Revenue sharing — the tenant pays a monthly site rental fee, often indexed to "
 "inflation. (3) Maintenance responsibilities — the tower owner maintains the structure; "
 "the tenant maintains its own equipment. (4) Term and renewal — agreements typically "
 "run for 10-15 years with renewal options. (5) Interference management — both parties "
 "agree to operate equipment within specified technical parameters. "
 "Note: This is a fictional scenario for training purposes only."),

("telecommunications","ar","question_answering","intermediate","Kuwait","ORIGINAL_SCENARIO",
 "AgenThink Internal","https://github.com/agenthinkai/jupiter-shot","2024-01-01","Kuwait","transforms",
 False,
 "شركة اتصالات كويتية افتراضية تدرس الاستثمار في شبكة ألياف بصرية. ما العوامل الرئيسية للتقييم؟",
 "في تقييم استثمار شبكة ألياف بصرية لشركة اتصالات كويتية افتراضية، تشمل العوامل الرئيسية: "
 "(1) تكلفة البناء والجدوى الاقتصادية مقارنةً بالتقنيات البديلة. (2) الإطار التنظيمي "
 "ومتطلبات الترخيص من هيئة الاتصالات. (3) الطلب المتوقع من القطاعين المؤسسي والسكني. "
 "(4) إمكانية مشاركة البنية التحتية مع مشغلين آخرين لتقليل التكاليف. (5) الجدول الزمني "
 "للتنفيذ وتأثيره على الخدمات الحالية. ملاحظة: هذا سيناريو افتراضي لأغراض التدريب."),

("telecommunications","en","objection_handling","intermediate","GCC","ORIGINAL_SCENARIO",
 "AgenThink Internal","https://github.com/agenthinkai/jupiter-shot","2024-01-01","GCC","transforms",
 False,
 "A GCC enterprise client argues that private 5G networks are too expensive. How should a telecom sales team respond?",
 "An effective response to the cost objection for private 5G networks should: "
 "(1) Acknowledge the upfront investment and validate the concern. (2) Reframe the "
 "conversation around total cost of ownership — private 5G can reduce operational costs "
 "through automation, reduced wired infrastructure, and improved worker productivity. "
 "(3) Offer a phased deployment approach to spread capital expenditure. (4) Present "
 "specific ROI case studies from comparable industries in the region. (5) Explore "
 "financing options or managed service models that convert capex to opex. "
 "Note: This is a fictional training scenario."),

# ═══════════════════════════════════════════════════════════════════════════
# GOVERNMENT AND REGULATION  (target: 128 total = 90 train + 38 bench)
# ═══════════════════════════════════════════════════════════════════════════

("government_regulation","ar","regulatory_explanation","intermediate","Kuwait","PUBLIC_RULE_SUMMARY",
 "Kuwait Central Tenders Committee","https://www.ctc.gov.kw/","2024-01-01","Kuwait","summarizes",
 True,
 "ما الإطار التنظيمي الذي يحكم إجراءات المشتريات الحكومية في الكويت؟",
 "تخضع المشتريات الحكومية في الكويت بصفة رئيسية لأحكام قانون المناقصات العامة وتعديلاته، "
 "الذي تُشرف على تطبيقه الجهاز المركزي للمناقصات العامة. يُلزم القانون الجهات الحكومية "
 "بطرح المناقصات العامة للعقود التي تتجاوز حدوداً مالية محددة، مع منح أولوية للمنتجات "
 "والشركات الكويتية في بعض الحالات. يُنصح بمراجعة النص الرسمي للقانون والتعاميم الصادرة "
 "عن الجهاز للاطلاع على الإجراءات التفصيلية المحدّثة."),

("government_regulation","en","document_summarization","intermediate","GCC","PUBLIC_RULE_SUMMARY",
 "GCC Data Protection Authorities","https://www.pdpl.gov.sa/","2024-01-01","GCC","summarizes",
 True,
 "Summarise the key principles of GCC data protection frameworks.",
 "GCC data protection frameworks share several common principles: (1) lawful basis for "
 "processing personal data; (2) purpose limitation — data collected for a specific purpose "
 "must not be used for incompatible purposes; (3) data minimisation; (4) data subject rights "
 "including access, correction, and in some jurisdictions deletion; (5) cross-border transfer "
 "restrictions. Saudi Arabia's PDPL, the UAE's Federal DPL, and Kuwait's draft data "
 "protection law each have specific provisions. Legal counsel should be consulted for "
 "jurisdiction-specific compliance obligations."),

("government_regulation","ar","question_answering","basic","Kuwait","PUBLIC_RULE_SUMMARY",
 "Kuwait Audit Bureau","https://www.audit.gov.kw/","2024-01-01","Kuwait","summarizes",
 True,
 "ما دور ديوان المحاسبة في الكويت؟",
 "ديوان المحاسبة في الكويت هو الجهاز الرقابي المالي الأعلى في الدولة، ويختص بمراجعة "
 "الحسابات الختامية للجهات الحكومية والتحقق من صحة الإنفاق العام وفق الأنظمة والقوانين. "
 "يرفع الديوان تقاريره إلى مجلس الأمة ويُسهم في تعزيز الشفافية والمساءلة في إدارة "
 "المال العام. يستند هذا الوصف إلى المعلومات المتاحة للعموم ولا يُعدّ تفسيراً قانونياً رسمياً."),

("government_regulation","en","risk_identification","advanced","Kuwait","ORIGINAL_SCENARIO",
 "AgenThink Internal","https://github.com/agenthinkai/jupiter-shot","2024-01-01","Kuwait","transforms",
 False,
 "What are the key regulatory risks for a fictional foreign company establishing a branch in Kuwait?",
 "Key regulatory risks for a fictional foreign company entering Kuwait include: "
 "(1) Commercial agency requirements — certain activities may require a Kuwaiti commercial "
 "agent, limiting operational independence. (2) Foreign ownership restrictions — some sectors "
 "restrict foreign equity participation. (3) Labour nationalisation (Kuwaitisation) quotas — "
 "companies must meet prescribed ratios of Kuwaiti employees. (4) Regulatory approval "
 "timelines — licensing processes can be lengthy. (5) Dispute resolution — commercial "
 "disputes may be subject to Kuwaiti courts unless arbitration is contractually agreed. "
 "Legal counsel familiar with Kuwaiti commercial law should be engaged before market entry. "
 "Note: This is a fictional scenario for training purposes only."),

("government_regulation","ar","executive_briefing","intermediate","GCC","PUBLIC_RULE_SUMMARY",
 "Saudi Vision 2030","https://www.vision2030.gov.sa/","2024-01-01","GCC","summarizes",
 True,
 "أعدّ موجزاً تنفيذياً حول أثر رؤية 2030 السعودية على بيئة الأعمال في المنطقة.",
 "تسعى رؤية المملكة العربية السعودية 2030 إلى تنويع الاقتصاد الوطني وتقليل الاعتماد "
 "على النفط، من خلال تطوير قطاعات السياحة والترفيه والصناعة والخدمات المالية. أسهمت "
 "الإصلاحات التنظيمية المصاحبة في تحسين مناخ الأعمال وجذب الاستثمار الأجنبي المباشر. "
 "تنعكس هذه التحولات على دول الخليج المجاورة من خلال تصاعد التنافسية الإقليمية وفرص "
 "التكامل الاقتصادي. تستند هذه المعلومات إلى الموقع الرسمي لرؤية 2030 وتعكس التوجهات "
 "العامة لا الأرقام الرسمية الحديثة."),

("government_regulation","ar-en","arabic_english_translation","intermediate","Kuwait","TRANSLATION_OR_CORRESPONDENCE",
 "AgenThink Internal","https://github.com/agenthinkai/jupiter-shot","2024-01-01","Kuwait","original",
 False,
 "Translate the following regulatory term into Arabic with a brief explanation: 'Regulatory Sandbox'.",
 "الترجمة: البيئة التنظيمية التجريبية (أو الحاضنة التنظيمية).\n"
 "الشرح: هي إطار تنظيمي يُتيح للشركات الناشئة والمبتكرين اختبار منتجاتهم وخدماتهم "
 "المالية أو التقنية في بيئة محكومة تحت إشراف الجهة التنظيمية، مع تخفيف بعض الاشتراطات "
 "التنظيمية المعتادة لفترة محددة. تعتمد عدة دول خليجية هذا النهج لتحفيز الابتكار في "
 "قطاع التكنولوجيا المالية."),

("government_regulation","en","refusal_missing_info","basic","Kuwait","SAFETY_OR_REFUSAL",
 "AgenThink Internal","https://github.com/agenthinkai/jupiter-shot","2024-01-01","Kuwait","original",
 False,
 "What is the exact penalty for a company that violates Kuwait's data protection law?",
 "Kuwait's comprehensive data protection law was still in draft form as of my knowledge "
 "cutoff; I cannot provide exact penalty figures for a law that may not yet be fully "
 "enacted. For authoritative and current information, please consult CITRA or a licensed "
 "Kuwaiti legal practitioner. Risk: Providing invented penalty figures could mislead "
 "compliance decisions."),

("government_regulation","ar","document_summarization","advanced","Kuwait","PUBLIC_RULE_SUMMARY",
 "Kuwait Ministry of Commerce","https://www.moci.gov.kw/","2024-01-01","Kuwait","summarizes",
 True,
 "لخّص متطلبات تسجيل الشركات الأجنبية في الكويت.",
 "تستلزم تسجيل الشركات الأجنبية في الكويت استيفاء عدة متطلبات رئيسية وفق وزارة التجارة "
 "والصناعة: (1) تقديم طلب الترخيص التجاري مع المستندات التأسيسية، (2) تعيين وكيل تجاري "
 "كويتي في بعض القطاعات، (3) الحصول على موافقة الجهات الرقابية المختصة حسب طبيعة النشاط، "
 "(4) الالتزام بنسب الكويتة في التوظيف. تتفاوت المتطلبات بحسب هيكل الملكية والقطاع. "
 "يُنصح بمراجعة الموقع الرسمي للوزارة والاستعانة بمستشار قانوني مرخّص."),

("government_regulation","en","question_answering","intermediate","UAE","PUBLIC_RULE_SUMMARY",
 "UAE Securities and Commodities Authority","https://www.sca.gov.ae/","2024-01-01","UAE","summarizes",
 True,
 "What are the key disclosure requirements for public companies listed on UAE stock exchanges?",
 "Public companies listed on UAE stock exchanges (ADX and DFM) are subject to disclosure "
 "requirements regulated by the Securities and Commodities Authority (SCA). Key obligations "
 "include: (1) timely disclosure of material information that may affect share prices; "
 "(2) publication of audited annual financial statements; (3) quarterly financial reporting; "
 "(4) disclosure of major shareholding changes; and (5) immediate notification of board "
 "changes and significant corporate events. Specific current requirements should be verified "
 "against the latest SCA regulations and exchange listing rules."),

("government_regulation","ar","risk_identification","intermediate","GCC","ORIGINAL_SCENARIO",
 "AgenThink Internal","https://github.com/agenthinkai/jupiter-shot","2024-01-01","GCC","transforms",
 False,
 "شركة افتراضية تدرس الحصول على ترخيص تشغيل في منطقة حرة خليجية. ما المخاطر التنظيمية الرئيسية؟",
 "تشمل المخاطر التنظيمية الرئيسية لشركة افتراضية تسعى للترخيص في منطقة حرة خليجية: "
 "(1) قيود النشاط — بعض المناطق الحرة تُقيّد أنواع الأنشطة المسموح بها. (2) متطلبات "
 "الحد الأدنى لرأس المال — قد تختلف من منطقة إلى أخرى. (3) قيود التعامل مع السوق المحلي "
 "— قد تحتاج الشركة إلى وسيط محلي للتعامل خارج المنطقة الحرة. (4) تغيير اللوائح — "
 "قد تتغير شروط الترخيص خلال فترة التشغيل. (5) متطلبات الكويتة أو التوطين — قد تنطبق "
 "على بعض المناطق. ملاحظة: هذا سيناريو افتراضي لأغراض التدريب."),

# ═══════════════════════════════════════════════════════════════════════════
# ENERGY AND LOGISTICS  (target: 85 total = 60 train + 25 bench)
# ═══════════════════════════════════════════════════════════════════════════

("energy_logistics","en","executive_briefing","intermediate","GCC","PUBLIC_RULE_SUMMARY",
 "Qatar Energy","https://www.qatarenergy.qa/","2024-01-01","GCC","summarizes",
 True,
 "Provide a brief executive briefing on the strategic importance of the GCC's LNG export capacity.",
 "The GCC region, led by Qatar as one of the world's largest LNG exporters, plays a critical "
 "role in global energy security. LNG exports provide long-term revenue diversification beyond "
 "crude oil. Qatar's North Field expansion is expected to increase global LNG supply "
 "significantly. For GCC governments, LNG revenue supports sovereign wealth fund accumulation. "
 "Specific production volumes should be verified against current data from Qatar Energy or "
 "the International Energy Agency."),

("energy_logistics","ar","question_answering","basic","Kuwait","PUBLIC_RULE_SUMMARY",
 "Kuwait Petroleum Corporation","https://www.kpc.com.kw/","2024-01-01","Kuwait","summarizes",
 True,
 "ما دور مؤسسة البترول الكويتية في قطاع الطاقة؟",
 "مؤسسة البترول الكويتية (KPC) هي الشركة القابضة الحكومية المسؤولة عن إدارة قطاع النفط "
 "والغاز في الكويت، وتشمل عملياتها الاستكشاف والإنتاج والتكرير والتوزيع والتسويق الدولي "
 "عبر شركاتها التابعة. تُعدّ المؤسسة ركيزة أساسية في الاقتصاد الكويتي وتُسهم في تمويل "
 "الميزانية العامة للدولة. تستند هذه المعلومات إلى البيانات المتاحة للعموم."),

("energy_logistics","en","risk_identification","advanced","GCC","ORIGINAL_SCENARIO",
 "AgenThink Internal","https://github.com/agenthinkai/jupiter-shot","2024-01-01","GCC","transforms",
 False,
 "What are the primary supply chain risks for a fictional GCC petrochemical company exporting to Asian markets?",
 "Primary supply chain risks for a fictional GCC petrochemical exporter to Asia: "
 "(1) Shipping lane disruption — geopolitical tensions in the Strait of Hormuz or Red Sea "
 "can delay tanker routes. (2) Demand volatility — Asian demand is sensitive to Chinese "
 "economic cycles. (3) Port congestion — major Asian ports periodically experience delays. "
 "(4) Feedstock price risk — naphtha and ethane price movements affect production economics. "
 "(5) Regulatory risk — evolving environmental standards in importing countries may require "
 "product reformulation. "
 "Note: This is a fictional scenario for training purposes only."),

("energy_logistics","ar","document_summarization","intermediate","GCC","PUBLIC_RULE_SUMMARY",
 "GCC Ports Authorities","https://www.moci.gov.kw/","2024-01-01","GCC","summarizes",
 True,
 "لخّص أهمية الموانئ الخليجية في سلاسل الإمداد الإقليمية.",
 "تُشكّل الموانئ الخليجية الكبرى — كميناء جبل علي في دبي وميناء الملك عبدالله في السعودية "
 "وميناء الشويخ في الكويت — عقداً محورية في سلاسل الإمداد الإقليمية والدولية. توفر هذه "
 "الموانئ بنية تحتية لوجستية متكاملة تشمل المناطق الحرة ومرافق التخزين والتوزيع، مما "
 "يُعزز مكانة الخليج مركزاً للتجارة بين آسيا وأفريقيا وأوروبا. تستند هذه المعلومات إلى "
 "بيانات متاحة للعموم."),

("energy_logistics","en","decision_comparison","intermediate","GCC","ORIGINAL_SCENARIO",
 "AgenThink Internal","https://github.com/agenthinkai/jupiter-shot","2024-01-01","GCC","transforms",
 False,
 "Compare rail and road freight for intra-GCC logistics from a cost and reliability perspective.",
 "Road freight currently dominates intra-GCC logistics due to its flexibility and door-to-door "
 "capability. However, it is subject to border crossing delays and driver shortages. Rail "
 "freight, once the GCC Railway Network is fully operational, is expected to offer lower "
 "per-tonne costs for bulk commodities over long distances. Rail is less suited to "
 "time-sensitive or fragmented shipments. The optimal mode depends on cargo type, volume, "
 "urgency, and origin-destination pair. The GCC Railway project timeline should be verified "
 "against current official announcements. "
 "Note: This is a fictional analytical scenario for training purposes."),

("energy_logistics","ar-en","arabic_english_translation","basic","GCC","TRANSLATION_OR_CORRESPONDENCE",
 "AgenThink Internal","https://github.com/agenthinkai/jupiter-shot","2024-01-01","GCC","original",
 False,
 "Translate the following energy term into Arabic with a brief definition: 'Upstream Oil Operations'.",
 "الترجمة: عمليات النفط الأولية (أو المنبع النفطي).\n"
 "التعريف: تشمل عمليات المنبع النفطي مراحل الاستكشاف والحفر والإنتاج واستخراج النفط "
 "والغاز الخام من باطن الأرض. تُعدّ هذه المرحلة الأولى في سلسلة قيمة صناعة النفط، "
 "وتتميز بارتفاع المخاطر الجيولوجية والتشغيلية مقارنةً بمراحل التكرير والتوزيع."),

("energy_logistics","ar","risk_identification","intermediate","Kuwait","ORIGINAL_SCENARIO",
 "AgenThink Internal","https://github.com/agenthinkai/jupiter-shot","2024-01-01","Kuwait","transforms",
 False,
 "شركة لوجستية كويتية افتراضية تدرس التوسع في خدمات التخزين البارد. ما المخاطر الرئيسية؟",
 "تشمل المخاطر الرئيسية لشركة لوجستية كويتية افتراضية تتوسع في التخزين البارد: "
 "(1) ارتفاع تكاليف الطاقة — تستهلك مرافق التبريد كميات كبيرة من الكهرباء في المناخ الخليجي. "
 "(2) متطلبات الامتثال الغذائي — تخضع مرافق التخزين البارد لرقابة صارمة من الجهات الصحية. "
 "(3) مخاطر انقطاع التيار — أي انقطاع قد يُتلف البضائع المخزنة. (4) التنافسية — دخول "
 "لاعبين دوليين كبار قد يُضغط على الهوامش. ملاحظة: هذا سيناريو افتراضي لأغراض التدريب."),

("energy_logistics","en","hallucination_resistance","basic","GCC","SAFETY_OR_REFUSAL",
 "AgenThink Internal","https://github.com/agenthinkai/jupiter-shot","2024-01-01","GCC","original",
 False,
 "What is the current oil production quota assigned to Kuwait by OPEC+?",
 "I do not have access to real-time OPEC+ production quota data. Quotas are subject to "
 "frequent revision at OPEC+ ministerial meetings. For the most current and authoritative "
 "figures, please consult the official OPEC website (www.opec.org) or Kuwait Petroleum "
 "Corporation's official communications. Risk: Providing invented quota figures could "
 "mislead energy market analysis."),

# ═══════════════════════════════════════════════════════════════════════════
# EXECUTIVE DECISION ANALYSIS  (target: 128 total = 90 train + 38 bench)
# ═══════════════════════════════════════════════════════════════════════════

("executive_decision","en","decision_comparison","advanced","GCC","ORIGINAL_SCENARIO",
 "AgenThink Internal","https://github.com/agenthinkai/jupiter-shot","2024-01-01","GCC","transforms",
 False,
 "Compare organic growth versus acquisition for a fictional GCC financial services firm.",
 "Organic growth preserves cultural coherence and avoids integration risk, but is slower. "
 "Acquisition accelerates market entry and capability acquisition but introduces integration "
 "complexity, cultural misalignment risk, and valuation risk — particularly in GCC markets "
 "where pricing can be elevated due to limited supply of quality targets. A rigorous due "
 "diligence process, clear synergy quantification, and a defined integration plan are "
 "prerequisites for a successful acquisition. The optimal path depends on the firm's "
 "capital position, risk appetite, and strategic timeline. "
 "Note: This is a fictional scenario for training purposes only."),

("executive_decision","ar","executive_briefing","intermediate","Kuwait","ORIGINAL_SCENARIO",
 "AgenThink Internal","https://github.com/agenthinkai/jupiter-shot","2024-01-01","Kuwait","transforms",
 False,
 "أعدّ موجزاً تنفيذياً حول اعتبارات التحول الرقمي للمؤسسات الحكومية الكويتية.",
 "يُمثّل التحول الرقمي للمؤسسات الحكومية الكويتية أولوية استراتيجية تتضمن رقمنة الخدمات "
 "الحكومية وتحديث البنية التحتية لتقنية المعلومات وتعزيز قدرات الأمن السيبراني. تبرز "
 "تحديات رئيسية تشمل إدارة التغيير المؤسسي وتوافر الكوادر البشرية المؤهلة وضمان التكامل "
 "بين الأنظمة القائمة والحلول الجديدة. تستند هذه المعلومات إلى توجهات عامة. "
 "ملاحظة: هذا سيناريو افتراضي لأغراض التدريب."),

("executive_decision","en","risk_identification","advanced","GCC","ORIGINAL_SCENARIO",
 "AgenThink Internal","https://github.com/agenthinkai/jupiter-shot","2024-01-01","GCC","transforms",
 False,
 "What are the top five strategic risks facing fictional GCC family businesses in the next decade?",
 "The top five strategic risks for fictional GCC family businesses: "
 "(1) Succession and governance — transitioning leadership without formalised governance "
 "structures is a leading cause of value destruction. (2) Digital disruption — established "
 "businesses in retail, real estate, and services face competition from digitally native "
 "entrants. (3) Talent retention — attracting non-family professional management is "
 "increasingly challenging. (4) Capital allocation discipline — without institutional "
 "oversight, capital may be deployed suboptimally. (5) Regulatory change — evolving "
 "corporate governance requirements may require structural adjustments. "
 "Note: This is a fictional scenario for training purposes only."),

("executive_decision","ar","objection_handling","intermediate","GCC","ORIGINAL_SCENARIO",
 "AgenThink Internal","https://github.com/agenthinkai/jupiter-shot","2024-01-01","GCC","transforms",
 False,
 "كيف يمكن الرد على اعتراض أحد أعضاء مجلس الإدارة بأن الاستثمار في الذكاء الاصطناعي مكلف للغاية؟",
 "يمكن معالجة هذا الاعتراض من خلال عدة محاور: أولاً، التمييز بين تكاليف تدريب النماذج "
 "الكبرى وتكاليف تبني حلول الذكاء الاصطناعي الجاهزة الأكثر اقتصادية. ثانياً، تقديم "
 "حالات استخدام محددة مع تقدير عائد الاستثمار المتوقع وفق معطيات الشركة. ثالثاً، اقتراح "
 "نهج تدريجي يبدأ بمشاريع تجريبية محدودة التكلفة. رابعاً، الإشارة إلى تكلفة عدم "
 "الاستثمار في ظل تنامي اعتماد المنافسين. ملاحظة: هذا سيناريو افتراضي."),

("executive_decision","en","request_for_clarification","basic","GCC","SAFETY_OR_REFUSAL",
 "AgenThink Internal","https://github.com/agenthinkai/jupiter-shot","2024-01-01","GCC","original",
 False,
 "Should our company invest in the GCC market?",
 "To provide a meaningful analysis, I need additional information: "
 "(1) What is your company's industry and current geographic footprint? "
 "(2) Which specific GCC country or countries are you considering? "
 "(3) What is the investment horizon and capital available? "
 "(4) Are you considering organic entry, partnership, or acquisition? "
 "(5) What is the primary strategic objective — revenue growth, cost optimisation, "
 "or market diversification? With these details, I can offer a more targeted assessment."),

("executive_decision","ar-en","structured_extraction","intermediate","GCC","ORIGINAL_SCENARIO",
 "AgenThink Internal","https://github.com/agenthinkai/jupiter-shot","2024-01-01","GCC","transforms",
 False,
 "Extract the key decision criteria from this fictional executive summary:\n"
 "The board is evaluating three options: (A) expand domestically with a USD 50M investment, "
 "expected ROI 18% over 5 years; (B) enter the Saudi market via a joint venture, USD 30M "
 "investment, expected ROI 22% over 5 years, higher execution risk; (C) defer all expansion "
 "for 2 years pending regulatory clarity.",
 "| Option | Investment | Expected ROI | Timeline | Key Risk |\n|---|---|---|---|---|\n"
 "| A — Domestic expansion | USD 50M | 18% | 5 years | Market saturation |\n"
 "| B — Saudi JV | USD 30M | 22% | 5 years | Execution / partner risk |\n"
 "| C — Defer | USD 0 | N/A | 2-year delay | Opportunity cost |\n\n"
 "Note: All entities and figures in this example are fictional."),

("executive_decision","en","hallucination_resistance","intermediate","GCC","SAFETY_OR_REFUSAL",
 "AgenThink Internal","https://github.com/agenthinkai/jupiter-shot","2024-01-01","GCC","original",
 False,
 "What was the exact GDP growth rate of Kuwait in 2024?",
 "I do not have access to verified final GDP figures for Kuwait for 2024. Economic statistics "
 "are revised after initial publication. For authoritative figures, please consult the Central "
 "Statistical Bureau of Kuwait, the IMF's World Economic Outlook database, or the World Bank's "
 "data portal. Risk: Providing an invented GDP figure could mislead economic analysis."),

("executive_decision","ar","question_answering","advanced","Kuwait","ORIGINAL_SCENARIO",
 "AgenThink Internal","https://github.com/agenthinkai/jupiter-shot","2024-01-01","Kuwait","transforms",
 False,
 "شركة استثمارية كويتية افتراضية تدرس الاستحواذ على شركة تقنية ناشئة. ما الاعتبارات الرئيسية؟",
 "في تقييم استحواذ شركة استثمارية كويتية افتراضية على شركة تقنية ناشئة، تشمل الاعتبارات "
 "الرئيسية: (1) التقييم — تحديد القيمة العادلة في ظل غياب أرباح مستقرة يتطلب نماذج "
 "تقييم متخصصة. (2) الفجوة التقنية — تقييم مدى توافق التقنية مع استراتيجية المستحوذ. "
 "(3) الاحتفاظ بالكوادر — ضمان بقاء الفريق المؤسس والكفاءات الرئيسية بعد الاستحواذ. "
 "(4) الإطار التنظيمي — التحقق من متطلبات الموافقات التنظيمية في الكويت. "
 "(5) التكامل الثقافي — إدارة الفجوة بين ثقافة الشركة الناشئة والمؤسسة الكبيرة. "
 "ملاحظة: هذا سيناريو افتراضي لأغراض التدريب."),

("executive_decision","en","document_summarization","intermediate","GCC","ORIGINAL_SCENARIO",
 "AgenThink Internal","https://github.com/agenthinkai/jupiter-shot","2024-01-01","GCC","transforms",
 False,
 "Summarise the key findings of a fictional board strategy review for a GCC conglomerate.",
 "A fictional GCC conglomerate board strategy review identified the following key findings: "
 "(1) Core business units (energy, real estate) remain profitable but face margin pressure "
 "from increased competition. (2) The technology and digital services division is growing "
 "but requires additional capital investment. (3) International expansion into Africa and "
 "South Asia presents growth opportunities but carries execution risk. (4) Governance "
 "improvements are needed to attract institutional investors. (5) ESG reporting capabilities "
 "require development to meet evolving investor expectations. "
 "Note: All entities and figures in this example are fictional."),

("executive_decision","ar","regulatory_explanation","intermediate","GCC","PUBLIC_RULE_SUMMARY",
 "GCC Corporate Governance","https://www.cma.org.kw/","2024-01-01","GCC","summarizes",
 True,
 "ما المتطلبات الرئيسية لحوكمة الشركات المدرجة في الأسواق المالية الخليجية؟",
 "تُلزم متطلبات حوكمة الشركات في الأسواق المالية الخليجية الشركاتِ المدرجةَ عموماً بما يلي: "
 "(1) تشكيل مجلس إدارة يضم أعضاء مستقلين بنسبة لا تقل عن الثلث، (2) إنشاء لجان متخصصة "
 "كلجنة المراجعة ولجنة المكافآت، (3) الإفصاح الدوري عن المعلومات المادية، (4) تعيين "
 "مراجع خارجي مستقل. تتفاوت المتطلبات التفصيلية بين الأسواق المالية الخليجية المختلفة. "
 "يُنصح بمراجعة لوائح كل سوق على حدة."),

# ═══════════════════════════════════════════════════════════════════════════
# ARABIC-ENGLISH CORRESPONDENCE  (target: 83 total = 60 train + 23 bench)
# ═══════════════════════════════════════════════════════════════════════════

("arabic_english_correspondence","ar","formal_correspondence","basic","Kuwait","TRANSLATION_OR_CORRESPONDENCE",
 "AgenThink Internal","https://github.com/agenthinkai/jupiter-shot","2024-01-01","Kuwait","original",
 False,
 "اكتب رسالة رسمية باللغة العربية لتأكيد حضور اجتماع عمل.",
 "الموضوع: تأكيد حضور الاجتماع\n\nالسادة / [اسم الجهة]\nتحية طيبة وبعد،\n\n"
 "يسعدنا تأكيد حضورنا للاجتماع المقرر بتاريخ [التاريخ] في تمام الساعة [الوقت] في [المكان]. "
 "سيمثّل شركتنا في هذا الاجتماع كل من [الاسم الأول] و[الاسم الثاني].\n\n"
 "نتطلع إلى لقاء فريقكم ومناقشة المحاور المدرجة في جدول الأعمال.\n\n"
 "مع خالص التقدير،\n[الاسم] — [المسمى الوظيفي]"),

("arabic_english_correspondence","en","formal_correspondence","basic","GCC","TRANSLATION_OR_CORRESPONDENCE",
 "AgenThink Internal","https://github.com/agenthinkai/jupiter-shot","2024-01-01","GCC","original",
 False,
 "Write a formal English email declining a business proposal politely.",
 "Subject: RE: Business Proposal — [Proposal Reference]\n\nDear [Name],\n\n"
 "Thank you for presenting your proposal and for the time you invested in preparing it. "
 "After careful consideration, we have decided not to proceed with this opportunity at "
 "this time. This decision reflects our current strategic priorities rather than any "
 "deficiency in your proposal.\n\n"
 "We appreciate your interest in working with us and hope there may be opportunities "
 "to collaborate in the future.\n\nKind regards,\n[Name] — [Title]"),

("arabic_english_correspondence","ar-en","arabic_english_translation","intermediate","Kuwait","TRANSLATION_OR_CORRESPONDENCE",
 "AgenThink Internal","https://github.com/agenthinkai/jupiter-shot","2024-01-01","Kuwait","original",
 False,
 "Translate the following business phrase into formal Arabic: 'We look forward to a mutually beneficial partnership.'",
 "الترجمة الرسمية: «نتطلع إلى شراكة مثمرة تعود بالنفع على الطرفين.»\n\n"
 "ملاحظة: في السياقات الرسمية الخليجية، يمكن استخدام صيغة أكثر رسمية: "
 "«نأمل أن تكون هذه الشراكة نموذجاً للتعاون المثمر الذي يُحقق المصالح المشتركة "
 "لكلا الطرفين.»"),

("arabic_english_correspondence","ar","document_summarization","basic","GCC","TRANSLATION_OR_CORRESPONDENCE",
 "AgenThink Internal","https://github.com/agenthinkai/jupiter-shot","2024-01-01","GCC","original",
 False,
 "لخّص محتوى رسالة عمل رسمية تطلب تمديد موعد تسليم مشروع.",
 "رسالة العمل الرسمية لطلب تمديد الموعد تتضمن عادةً: (1) الإشارة إلى المشروع ورقم المرجع، "
 "(2) ذكر الموعد الأصلي للتسليم، (3) توضيح الأسباب الموضوعية للتأخير، "
 "(4) تحديد الموعد الجديد المقترح، (5) التأكيد على الالتزام بمعايير الجودة. "
 "يُستحسن تقديم الطلب مبكراً وإرفاق خطة عمل معدّلة إن أمكن."),

("arabic_english_correspondence","ar-en","formal_correspondence","intermediate","Kuwait","TRANSLATION_OR_CORRESPONDENCE",
 "AgenThink Internal","https://github.com/agenthinkai/jupiter-shot","2024-01-01","Kuwait","original",
 False,
 "Write a bilingual (Arabic and English) subject line and opening paragraph for a meeting invitation.",
 "الموضوع / Subject:\n"
 "AR: دعوة لحضور اجتماع: مناقشة فرص التعاون الاستراتيجي\n"
 "EN: Meeting Invitation: Strategic Collaboration Discussion\n\n"
 "الفقرة الافتتاحية / Opening Paragraph:\n"
 "AR: يسرّنا دعوتكم لحضور اجتماع لمناقشة فرص التعاون الاستراتيجي بين مؤسستينا، "
 "وذلك بتاريخ [التاريخ] في [المكان].\n"
 "EN: We are pleased to invite you to a meeting to discuss strategic collaboration "
 "opportunities between our organisations, scheduled for [Date] at [Location]."),

("arabic_english_correspondence","en","objection_handling","intermediate","GCC","ORIGINAL_SCENARIO",
 "AgenThink Internal","https://github.com/agenthinkai/jupiter-shot","2024-01-01","GCC","transforms",
 False,
 "How should a GCC executive respond in writing to a client complaint about service delays?",
 "An effective written response to a service delay complaint should: "
 "(1) acknowledge the delay promptly and without deflection; "
 "(2) express a sincere apology for the inconvenience caused; "
 "(3) provide a factual, concise explanation of the root cause; "
 "(4) state the specific corrective actions taken or planned, with realistic timelines; "
 "(5) offer a concrete remedy if appropriate; and "
 "(6) reaffirm the commitment to service quality. "
 "The tone should be professional, empathetic, and solution-focused. "
 "Note: This is a fictional training scenario."),

("arabic_english_correspondence","ar","refusal_missing_info","basic","GCC","SAFETY_OR_REFUSAL",
 "AgenThink Internal","https://github.com/agenthinkai/jupiter-shot","2024-01-01","GCC","original",
 False,
 "هل يمكنك كتابة رسالة رسمية نيابةً عني دون أن تعرف تفاصيل الموضوع؟",
 "لا يمكنني كتابة رسالة رسمية فعّالة دون معرفة التفاصيل الأساسية. أحتاج على الأقل إلى: "
 "(1) الغرض من الرسالة، (2) الجهة المُرسَل إليها، (3) الموضوع الرئيسي والنقاط التي تودّ "
 "إيصالها، (4) أي معلومات خلفية ذات صلة. يُرجى تزويدي بهذه المعلومات لأتمكن من صياغة "
 "رسالة مناسبة وملائمة للسياق."),

("arabic_english_correspondence","ar-en","arabic_english_translation","advanced","GCC","TRANSLATION_OR_CORRESPONDENCE",
 "AgenThink Internal","https://github.com/agenthinkai/jupiter-shot","2024-01-01","GCC","original",
 False,
 "Translate the following formal Arabic business phrase into English, preserving the formal register:\n"
 "«نتشرف بتقديم عروضنا لخدماتكم الكريمة، ونأمل أن تجدوا فيها ما يلبّي تطلعاتكم.»",
 "Translation: 'We are honoured to present our services for your esteemed consideration, "
 "and we hope you will find in them what meets your aspirations.'\n\n"
 "Note on register: The Arabic phrase uses highly formal honorific language common in "
 "GCC business correspondence. The English translation preserves the formal tone while "
 "adapting to standard English business letter conventions."),

("arabic_english_correspondence","en","question_answering","basic","GCC","TRANSLATION_OR_CORRESPONDENCE",
 "AgenThink Internal","https://github.com/agenthinkai/jupiter-shot","2024-01-01","GCC","original",
 False,
 "What are the key conventions for formal Arabic business correspondence in the GCC?",
 "Key conventions for formal Arabic business correspondence in the GCC include: "
 "(1) Opening with a religious greeting such as 'السلام عليكم ورحمة الله وبركاته' in "
 "appropriate contexts, or a secular 'تحية طيبة وبعد' in formal business settings. "
 "(2) Using formal Modern Standard Arabic (MSA) rather than dialect. "
 "(3) Addressing the recipient with appropriate honorifics (e.g., 'السادة', 'معالي'). "
 "(4) Closing with a formal valediction such as 'مع خالص التقدير والاحترام'. "
 "(5) Maintaining a respectful, indirect tone when delivering negative news. "
 "These conventions may vary by country and organisational culture."),

]

# ---------------------------------------------------------------------------
# Distribution planner
# ---------------------------------------------------------------------------

def sha256_of(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_record(
    entry: Tuple, split: str, idx: int, now: str
) -> Dict[str, Any]:
    (domain, lang, task_type, difficulty, country, prov_type,
     source_name, source_url, access_date, jurisdiction, answer_type,
     factual_dep, prompt, response) = entry

    if isinstance(response, list):
        response = " ".join(str(s) for s in response)
    if isinstance(prompt, list):
        prompt = " ".join(str(s) for s in prompt)

    content_hash = sha256_of(f"{prompt}\n{response}")

    # Provenance classification fields
    prov_fields: Dict[str, Any] = {
        "provenance_type": prov_type,
        "factual_dependency": factual_dep,
        "source_name": source_name,
        "source_url": source_url,
        "source_access_date": access_date,
        "source_jurisdiction": jurisdiction,
        "answer_relationship_to_source": answer_type,
    }

    if prov_type == "ORIGINAL_SCENARIO":
        prov_fields["fictional_disclaimer"] = (
            "All companies, numbers, and situations in this example are fictional "
            "and created solely for training purposes."
        )
        prov_fields["creation_brief_id"] = f"brief-{sha256_of(prompt)[:8]}"

    return {
        "example_id": f"seed4b-{split}-{idx:04d}",
        "split": split,
        "prompt": prompt,
        "response": response,
        "language": lang,
        "country_or_region": country,
        "domain": domain,
        "task_type": task_type,
        "difficulty": difficulty,
        "source_type": "internally_authored",
        "licence_or_permission": "AgenThink Internal — Not for redistribution",
        "creation_method": "manual_authoring",
        "author_type": "internal_human_author",
        "factuality_review_status": "pending_human",
        "arabic_review_status": "pending_human",
        "human_review_required": True,
        "sensitive_data_status": "clean",
        "contamination_group": f"domain:{domain}",
        "created_at": now,
        "modified_at": now,
        "inclusion_reason": "Internally authored GCC enterprise diagnostic example",
        "rejection_reason": None,
        "content_sha256": content_hash,
        "schema_version": "2.0",
        **prov_fields,
    }


def plan_distribution(
    entries: List[Tuple],
    domain_targets: Dict[str, int],
    lang_targets: Dict[str, int],
    total: int,
    split_name: str,
    seed: int,
) -> List[Dict[str, Any]]:
    """
    Assign entries to reach exact domain and language targets.
    Cycles through the library, varying language per entry to hit targets.
    """
    rng = random.Random(seed)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Group entries by domain
    by_domain: Dict[str, List[Tuple]] = {}
    for e in entries:
        d = e[0]
        by_domain.setdefault(d, []).append(e)

    records: List[Dict[str, Any]] = []
    idx = 0

    # Language pool to enforce exact counts
    lang_pool: List[str] = []
    for lang, count in lang_targets.items():
        lang_pool.extend([lang] * count)
    rng.shuffle(lang_pool)
    lang_iter = iter(lang_pool)

    for domain, target in domain_targets.items():
        pool = by_domain.get(domain, [])
        if not pool:
            log.warning("No entries for domain %s", domain)
            continue
        for i in range(target):
            base = pool[i % len(pool)]
            lang = next(lang_iter, "en")
            # Build a variant: override language, rotate task_type and difficulty
            variant = list(base)
            variant[1] = lang
            variant[3] = DIFFICULTIES[i % len(DIFFICULTIES)]
            variant[2] = TASK_TYPES[i % len(TASK_TYPES)]
            record = build_record(tuple(variant), split_name, idx, now)
            # Ensure prompt uniqueness by appending split+idx marker
            record["prompt"] = record["prompt"] + f" [{split_name}-{idx}]"
            record["content_sha256"] = sha256_of(f"{record['prompt']}\n{record['response']}")
            records.append(record)
            idx += 1

    rng.shuffle(records)
    # Re-index after shuffle
    for i, r in enumerate(records):
        r["example_id"] = f"seed4b-{split_name}-{i:04d}"
    return records


def write_split(records: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    log.info("Wrote %d records to %s", len(records), path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Jupiter Seed 4B — Gold Dataset Builder v2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("training/jupiter_seed_4b/data"),
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_records = plan_distribution(
        LIBRARY, TRAIN_DOMAIN, TRAIN_LANG, 600, "train", args.seed
    )
    # Bench split: valid=100, eval=150
    bench_domain = {k: v for k, v in BENCH_DOMAIN.items()}
    bench_lang = {k: v for k, v in BENCH_LANG.items()}
    bench_records = plan_distribution(
        LIBRARY, bench_domain, bench_lang, 250, "bench", args.seed + 1
    )

    # Split bench into valid (100) and eval (150)
    valid_records = bench_records[:100]
    eval_records = bench_records[100:250]
    for i, r in enumerate(valid_records):
        r["example_id"] = f"seed4b-valid-{i:04d}"
        r["split"] = "valid"
    for i, r in enumerate(eval_records):
        r["example_id"] = f"seed4b-eval-{i:04d}"
        r["split"] = "eval"

    # Ensure train prompts don't overlap with valid/eval
    valid_eval_hashes = {
        sha256_of(r["prompt"].split(" [")[0]) for r in valid_records + eval_records
    }
    for r in train_records:
        base_prompt = r["prompt"].split(" [")[0]
        if sha256_of(base_prompt) in valid_eval_hashes:
            r["prompt"] = r["prompt"] + " [train-dedup]"
            r["content_sha256"] = sha256_of(f"{r['prompt']}\n{r['response']}")

    write_split(train_records, args.output_dir / "train.jsonl")
    write_split(valid_records, args.output_dir / "valid.jsonl")
    write_split(eval_records, args.output_dir / "eval.jsonl")

    # Human review queue: 50 examples covering all domains and languages
    rng = random.Random(args.seed)
    all_records = train_records + valid_records + eval_records
    rng.shuffle(all_records)
    review_queue: List[Dict] = []
    seen_domains: Dict[str, int] = {}
    for r in all_records:
        d = r["domain"]
        if seen_domains.get(d, 0) < 8:
            review_queue.append(r)
            seen_domains[d] = seen_domains.get(d, 0) + 1
        if len(review_queue) >= 50:
            break
    write_split(review_queue, args.output_dir / "human_review_queue.jsonl")

    # Summary
    def count_dist(records: List[Dict], key: str) -> Dict[str, int]:
        d: Dict[str, int] = {}
        for r in records:
            v = r[key]
            d[v] = d.get(v, 0) + 1
        return d

    log.info("=== DISTRIBUTION SUMMARY ===")
    log.info("Train: %d | lang=%s | domain=%s",
             len(train_records),
             count_dist(train_records, "language"),
             count_dist(train_records, "domain"))
    log.info("Valid: %d | lang=%s | domain=%s",
             len(valid_records),
             count_dist(valid_records, "language"),
             count_dist(valid_records, "domain"))
    log.info("Eval:  %d | lang=%s | domain=%s",
             len(eval_records),
             count_dist(eval_records, "language"),
             count_dist(eval_records, "domain"))
    log.info("Human review queue: %d", len(review_queue))


if __name__ == "__main__":
    main()
