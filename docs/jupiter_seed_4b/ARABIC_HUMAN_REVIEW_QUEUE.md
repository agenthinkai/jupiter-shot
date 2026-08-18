# Jupiter Seed 4B — Arabic Human Review Queue

> **AI-ASSISTED CONTENT AUDIT — NOT HUMAN APPROVAL**
> This document is generated from the frozen corpus and Gate 14 findings.
> Human reviewers must independently judge every record.
> Software has not populated any reviewer judgment field.
> Package source commit: `50c20dd505e476c144ab433d05dc6d9a0d4c4bc6`
> Package identity scheme: `package_commit` identifies the source commit; the artifact release commit is its direct child.

## Coverage

| Metric | Value |
| :--- | :--- |
| Total records | 53 |
| Flagged (REVIEW_REQUIRED) | 9 |
| Not flagged (NONE) | 44 |
| Frozen base queue | 50 |
| Mandatory risk supplement | 3 |

## Flagged Records — Require Reviewer Attention

The following records were flagged by Gate 14 (Content-Risk).
Reviewers must explicitly judge each flagged record.

| Example ID | Field | Rule | Severity | Excerpt |
| :--- | :--- | :--- | :--- | :--- |
| `seed4b-eval-0004` | `response` | `GCC_TRIP_AR` | **REVIEW_REQUIRED** | هيكل الملكية والقطاع والنشاط. يُنصح بمراجعة الموقع الرسمي لوزارة التجارة والصناع |
| `seed4b-train-0015` | `response` | `GCC_TRIP_AR` | **REVIEW_REQUIRED** | ء الشرعي في التقارير السنوية. يُنصح بمراجعة أحدث التعاميم الصادرة عن البنك المرك |
| `seed4b-train-0016` | `response` | `GCC_TRIP_AR` | **REVIEW_REQUIRED** | والصناديق الإسلامية المدرجة. يُنصح بمراجعة الموقع الرسمي لساما للاطلاع على أحدث  |
| `seed4b-train-0028` | `response` | `GCC_TRIP_AR` | **REVIEW_REQUIRED** | لى برنامج مكافحة غسل الأموال. يُنصح بمراجعة أحدث التعاميم الصادرة عن المصرف المر |
| `seed4b-train-0032` | `response` | `GCC_TRIP_EN` | **REVIEW_REQUIRED** | ation employment commitments. Specific current requirements should be verified a |
| `seed4b-train-0038` | `response` | `GCC_TRIP_AR` | **REVIEW_REQUIRED** | وني وحماية بيانات المستهلكين. يُنصح بمراجعة الموقع الرسمي للهيئة للاطلاع على أحد |
| `seed4b-train-0039` | `response` | `GCC_TRIP_AR` | **REVIEW_REQUIRED** | الات وفق نظام التفضيل المحلي. يُنصح بمراجعة النص الرسمي للقانون والتعاميم الصادر |
| `seed4b-valid-0005` | `response` | `GCC_TRIP_AR` | **REVIEW_REQUIRED** | لا يجوز للمشغّل الاستئثار به. يُنصح بمراجعة المعيار الكامل والهيئة الشرعية للتطب |
| `seed4b-valid-0014` | `response` | `GCC_TRIP_AR` | **REVIEW_REQUIRED** | معقدة لإخفاء هويتهم الحقيقية. يُنصح بمراجعة التعاميم الصادرة عن البنك المركزي ال |

## All Queue Records

| # | Example ID | Split | Lang | Domain | Risk Status | Attention | Prompt (preview) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | `seed4b-valid-0001` | valid | en | islamic_finance | NONE | no | How does the IFSB approach liquidity risk management for Isl… |
| 2 | `seed4b-valid-0009` | valid | en | islamic_finance | NONE | no | Provide a brief executive briefing on the global Sukuk marke… |
| 3 | `seed4b-valid-0011` | valid | ar | gcc_banking | NONE | no | وفق التقارير السنوية لبنك الكويت المركزي، ما المؤشرات الرئيس… |
| 4 | `seed4b-train-0044` | train | ar-en | government_regulation | NONE | no | Translate the regulatory term 'Regulatory Sandbox' into Arab… |
| 5 | `seed4b-train-0032` | train | en | telecommunications | REVIEW_REQUIRED | **YES** | What are the key regulatory obligations for mobile network o… |
| 6 | `seed4b-train-0063` | train | ar | arabic_english_correspondence | NONE | no | لخّص العناصر الرئيسية التي يجب أن تتضمنها رسالة عمل رسمية تط… |
| 7 | `seed4b-train-0009` | train | ar-en | islamic_finance | NONE | no | Translate the Islamic finance term 'Sukuk' into Arabic and e… |
| 8 | `seed4b-train-0066` | train | ar | arabic_english_correspondence | NONE | no | هل يمكنك كتابة رسالة رسمية نيابةً عني دون أن تعرف تفاصيل الم… |
| 9 | `seed4b-train-0030` | train | ar | gcc_banking | NONE | no | استخرج المعلومات الرئيسية من ملخص التقرير الائتماني الافتراض… |
| 10 | `seed4b-train-0001` | train | ar | islamic_finance | NONE | no | ما تعريف عقد المشاركة في التمويل الإسلامي وكيف تُوزَّع الأرب… |
| 11 | `seed4b-valid-0010` | valid | en | gcc_banking | NONE | no | Explain the Net Stable Funding Ratio (NSFR) under Basel III … |
| 12 | `seed4b-train-0015` | train | ar | islamic_finance | REVIEW_REQUIRED | **YES** | ما الإطار التنظيمي الذي يحكم البنوك الإسلامية في الكويت وما … |
| 13 | `seed4b-valid-0005` | valid | ar | islamic_finance | REVIEW_REQUIRED | **YES** | لخّص المبادئ الأساسية للتأمين التكافلي وفق معايير هيئة أيوفي… |
| 14 | `seed4b-train-0010` | train | ar-en | islamic_finance | NONE | no | Translate 'Takaful' into Arabic and explain the three elemen… |
| 15 | `seed4b-eval-0016` | eval | ar | arabic_english_correspondence | NONE | no | هل يمكنك ترجمة وثيقتي إلى الإنجليزية دون أن أُرسل لك النص؟… |
| 16 | `seed4b-train-0042` | train | en | government_regulation | NONE | no | What are the key regulatory risks for a fictional foreign co… |
| 17 | `seed4b-train-0055` | train | ar | executive_decision | NONE | no | كيف يمكن الرد على اعتراض أحد أعضاء مجلس الإدارة بأن الاستثما… |
| 18 | `seed4b-train-0041` | train | ar | government_regulation | NONE | no | ما دور ديوان المحاسبة في الكويت وما صلاحياته الرقابية على ال… |
| 19 | `seed4b-train-0049` | train | ar | energy_logistics | NONE | no | لخّص أهمية الموانئ الخليجية الكبرى في سلاسل الإمداد الإقليمي… |
| 20 | `seed4b-eval-0011` | eval | ar | executive_decision | NONE | no | ما الاعتبارات الرئيسية لمعايير الحوكمة والمسؤولية الاجتماعية… |
| 21 | `seed4b-valid-0006` | valid | ar-en | islamic_finance | NONE | no | Extract the key risk factors from this fictional Mudaraba fu… |
| 22 | `seed4b-train-0056` | train | en | executive_decision | NONE | no | Should our company invest in the GCC market?… |
| 23 | `seed4b-train-0058` | train | en | executive_decision | NONE | no | What was the exact GDP growth rate of Kuwait in 2024?… |
| 24 | `seed4b-train-0021` | train | en | islamic_finance | NONE | no | A client argues that Islamic finance products are just conve… |
| 25 | `seed4b-train-0033` | train | ar | telecommunications | NONE | no | ما المقصود بجودة الخدمة في شبكات الاتصالات وما مؤشراتها التق… |
| 26 | `seed4b-valid-0012` | valid | ar-en | gcc_banking | NONE | no | Draft a short bilingual (Arabic and English) executive summa… |
| 27 | `seed4b-train-0065` | train | en | arabic_english_correspondence | NONE | no | How should a GCC executive respond in writing to a client co… |
| 28 | `seed4b-train-0052` | train | en | executive_decision | NONE | no | Compare organic growth versus acquisition as growth strategi… |
| 29 | `seed4b-eval-0000` | eval | ar | telecommunications | NONE | no | ما الفرق بين مشغل الشبكة المتنقلة الافتراضي (MVNO) ومشغل الش… |
| 30 | `seed4b-train-0026` | train | ar | gcc_banking | NONE | no | ما الفرق بين رأس المال من الشريحة الأولى والشريحة الثانية في… |
| 31 | `seed4b-train-0050` | train | ar | energy_logistics | NONE | no | شركة لوجستية كويتية افتراضية تدرس التوسع في خدمات التخزين ال… |
| 32 | `seed4b-eval-0009` | eval | ar-en | energy_logistics | NONE | no | Translate the energy term 'Upstream Oil Operations' into Ara… |
| 33 | `seed4b-train-0048` | train | en | energy_logistics | NONE | no | What are the primary supply chain risks for a fictional GCC … |
| 34 | `seed4b-train-0037` | train | en | telecommunications | NONE | no | What is the current 5G network coverage percentage in Kuwait… |
| 35 | `seed4b-train-0047` | train | ar | energy_logistics | NONE | no | ما دور مؤسسة البترول الكويتية في قطاع الطاقة الكويتي وما شرك… |
| 36 | `seed4b-train-0045` | train | en | government_regulation | NONE | no | What is the exact penalty for a company that violates Kuwait… |
| 37 | `seed4b-eval-0015` | eval | ar-en | arabic_english_correspondence | NONE | no | Translate this formal Arabic board resolution excerpt into p… |
| 38 | `seed4b-eval-0004` | eval | ar | government_regulation | REVIEW_REQUIRED | **YES** | ما متطلبات تسجيل الشركات الأجنبية في الكويت وفق وزارة التجار… |
| 39 | `seed4b-eval-0007` | eval | en | energy_logistics | NONE | no | What role are GCC sovereign wealth funds playing in energy t… |
| 40 | `seed4b-train-0036` | train | ar-en | telecommunications | NONE | no | Translate the telecom regulatory term 'Spectrum Auction' int… |
| 41 | `seed4b-train-0039` | train | ar | government_regulation | REVIEW_REQUIRED | **YES** | ما الإطار التنظيمي الذي يحكم إجراءات المشتريات الحكومية في ا… |
| 42 | `seed4b-train-0040` | train | en | government_regulation | NONE | no | Summarise the key principles common to GCC data protection f… |
| 43 | `seed4b-train-0067` | train | ar-en | arabic_english_correspondence | NONE | no | Translate this formal Arabic business phrase into English, p… |
| 44 | `seed4b-train-0062` | train | ar-en | arabic_english_correspondence | NONE | no | Translate the business phrase 'We look forward to a mutually… |
| 45 | `seed4b-train-0061` | train | en | arabic_english_correspondence | NONE | no | Write a formal English email politely declining a business p… |
| 46 | `seed4b-train-0034` | train | en | telecommunications | NONE | no | Identify the main commercial risks for a fictional GCC telec… |
| 47 | `seed4b-train-0046` | train | en | energy_logistics | NONE | no | Provide a brief executive briefing on the strategic importan… |
| 48 | `seed4b-train-0038` | train | ar | telecommunications | REVIEW_REQUIRED | **YES** | ما الإطار التنظيمي الذي يحكم قطاع الاتصالات في الإمارات العر… |
| 49 | `seed4b-train-0059` | train | ar | executive_decision | NONE | no | شركة استثمارية كويتية افتراضية تدرس الاستحواذ على شركة تقنية… |
| 50 | `seed4b-eval-0012` | eval | ar-en | executive_decision | NONE | no | Draft a short bilingual (Arabic and English) executive summa… |
| 51 | `seed4b-train-0016` | train | ar | islamic_finance | REVIEW_REQUIRED | **YES** | ما دور البنك المركزي السعودي ساما في تنظيم التمويل الإسلامي … |
| 52 | `seed4b-train-0028` | train | ar | gcc_banking | REVIEW_REQUIRED | **YES** | لخّص المتطلبات الرئيسية لمكافحة غسل الأموال المفروضة على الب… |
| 53 | `seed4b-valid-0014` | valid | ar | gcc_banking | REVIEW_REQUIRED | **YES** | ما المقصود بالملكية الفعلية (Beneficial Ownership) في إطار م… |

## Reviewer Instructions

For each record, complete the `REVIEWER_TEMPLATE.csv` columns:
- `reviewer_name`: your full name
- `accuracy_score`: 1–5
- `fluency_score`: 1–5
- `gcc_appropriateness_score`: 1–5
- `domain_terminology_score`: 1–5
- `factuality_score`: 1–5
- `verdict`: ACCEPT / REVISE / REJECT
- `corrected_wording`: if REVISE
- `rejection_reason`: if REJECT
- `comments`: optional

**Flagged records** (`REVIEW_REQUIRED`) require explicit judgment on the flagged phrase.

Software must never populate these fields.

---

Generated by `generate_review_package.py` — AI-ASSISTED INTERNAL AUDIT
