import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell

nb = nbf.read('notebook/01_generate_overall_satisfaction.ipynb', as_version=4)
cells = nb.cells

# =====================================================================
# Section 5: Corpus Analysis
# =====================================================================
cells.append(new_markdown_cell(r"""## Section 5 — Corpus Analysis

Before writing any rules, we analyse the **review text corpus** (`comment_clean`) to find recurring
expressions of *overall* satisfaction/dissatisfaction, and to confirm which expressions are
**aspect-specific** (e.g. "món ngon" = food is tasty, "giá rẻ" = price is cheap — these describe a
single aspect and are *excluded*) versus **global** (e.g. "sẽ quay lại" = will come back,
"không quay lại nữa" = won't come back again — these describe the overall experience and are the
basis for the 14 rule categories).

This section performs:
1. Basic text normalization for analysis purposes (lowercase, whitespace collapse).
2. Frequent n-gram extraction to surface candidate phrases.
3. Targeted frequency counts for hand-picked "global satisfaction" signal phrases.
4. A negation check — how often a candidate phrase is preceded by "không"/"chẳng"/"chưa" (which can
   reverse its polarity)."""))

cells.append(new_code_cell(r'''def _basic_normalize(text):
    """Lowercase + collapse whitespace, used only for corpus exploration in Section 5."""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize('NFC', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


corpus = df['comment_clean'].fillna('').map(_basic_normalize)
print(f"Corpus size: {len(corpus)} reviews")
print(f"Total characters: {corpus.str.len().sum():,}")
print(f"Mean review length (chars): {corpus.str.len().mean():.1f}")
'''))

cells.append(new_markdown_cell(r"""### 5.1 Frequent word n-grams

A quick word-level n-gram frequency count (1-3 grams) over the whole corpus, restricted to
alphabetic Vietnamese tokens, gives a high-level view of the vocabulary before we go looking for
specific phrases."""))

cells.append(new_code_cell(r'''TOKEN_RE = re.compile(r"[a-zà-ỹ0-9]+", re.IGNORECASE)

def tokenize(text):
    return TOKEN_RE.findall(text)

unigrams = Counter()
bigrams = Counter()
trigrams = Counter()
for text in corpus:
    toks = tokenize(text)
    unigrams.update(toks)
    bigrams.update(' '.join(p) for p in zip(toks, toks[1:]))
    trigrams.update(' '.join(p) for p in zip(toks, toks[1:], toks[2:]))

print("Top 15 unigrams:")
for w, c in unigrams.most_common(15):
    print(f"  {c:5d}  {w}")

print("\nTop 15 bigrams:")
for w, c in bigrams.most_common(15):
    print(f"  {c:5d}  {w}")

print("\nTop 15 trigrams:")
for w, c in trigrams.most_common(15):
    print(f"  {c:5d}  {w}")
'''))

cells.append(new_markdown_cell(r"""### 5.2 Aspect-specific vs. global expressions

We contrast two groups of phrases:

- **Aspect-specific** phrases describe a single aspect (food quality, price, atmosphere, service) and
  are intentionally **excluded** from the global `overall_satisfaction` rules, because they are
  already captured by `food_score` / `price_score` / `atmosphere_score` / `service_score`.
- **Global satisfaction** phrases describe the reviewer's *overall* experience or future
  behaviour (will they come back? would they recommend it? are they fully satisfied/disappointed
  overall?) — these are the candidates for the 14 rule categories."""))

cells.append(new_code_cell(r'''def phrase_count(phrase):
    pat = re.compile(re.escape(phrase))
    return sum(1 for t in corpus if pat.search(t))

aspect_specific_examples = ['món ngon', 'giá rẻ', 'giá hợp lý', 'đồ ăn ngon', 'phục vụ nhiệt tình',
                             'không gian đẹp', 'nhân viên thân thiện']
global_satisfaction_examples = ['sẽ quay lại', 'không quay lại', 'recommend', 'giới thiệu',
                                 'hài lòng', 'thất vọng', 'đáng đồng tiền', 'ủng hộ',
                                 'chờ lâu', 'khách quen']

print("Aspect-specific phrases (EXCLUDED from overall_satisfaction rules):")
for p in aspect_specific_examples:
    print(f"  {phrase_count(p):5d}  '{p}'")

print("\nGlobal satisfaction phrases (CANDIDATES for overall_satisfaction rules):")
for p in global_satisfaction_examples:
    print(f"  {phrase_count(p):5d}  '{p}'")
'''))

cells.append(new_markdown_cell(r"""### 5.3 Negation check for candidate phrases

Vietnamese negation words (`không`, `chẳng`, `chưa`, informal `ko`) placed immediately before a
phrase can **reverse its polarity** (e.g. "sẽ **không** quay lại" = will **not** come back;
"**không** hài lòng" = **not** satisfied). For every global-satisfaction candidate phrase we count
how often it is immediately preceded by a negation word, which tells us whether the phrase needs a
negative-lookbehind guard in its regex pattern."""))

cells.append(new_code_cell(r'''NEGATION_WORDS = ['không', 'chẳng', 'chưa', 'ko']

def negation_preceded_count(phrase, window_words=1):
    """Count reviews where `phrase` is immediately preceded by a negation word."""
    neg_alt = '|'.join(NEGATION_WORDS)
    pat = re.compile(rf'(?:{neg_alt}) {re.escape(phrase)}')
    return sum(1 for t in corpus if pat.search(t))

candidates = ['quay lại', 'ăn lại', 'recommend', 'ủng hộ', 'hài lòng', 'thất vọng',
               'xuất sắc', 'tệ', 'nên thử', 'đáng thử', 'xứng đáng']

print(f"{'phrase':15s} {'total':>7s} {'preceded by negation':>22s}")
for p in candidates:
    total = phrase_count(p)
    neg = negation_preceded_count(p)
    print(f"{p:15s} {total:7d} {neg:22d}")
'''))

cells.append(new_markdown_cell(r"""### 5.4 Sample reviews containing candidate global-satisfaction phrases

Reading a handful of real reviews per candidate phrase confirms (a) the phrase indeed expresses an
*overall* judgement (not an aspect-specific one), and (b) reveals real negation patterns to guard
against (e.g. "sẽ **không** làm bạn thất vọng" = "won't disappoint you", which is **positive**
despite containing "thất vọng")."""))

cells.append(new_code_cell(r'''def sample_reviews(phrase, n=3):
    pat = re.compile(re.escape(phrase))
    matched = [t for t in corpus if pat.search(t)]
    rng = np.random.RandomState(RANDOM_SEED)
    idx = rng.choice(len(matched), size=min(n, len(matched)), replace=False)
    return [matched[i] for i in idx]

for phrase in ['sẽ quay lại', 'không quay lại', 'thất vọng', 'xứng đáng']:
    print(f"--- '{phrase}' ---")
    for s in sample_reviews(phrase, n=2):
        snippet = s[:160] + ('...' if len(s) > 160 else '')
        print(f"  {snippet}")
    print()
'''))

# =====================================================================
# Section 6: Generate Rule Candidates
# =====================================================================
cells.append(new_markdown_cell(r"""## Section 6 — Generate Rule Candidates

Based on the corpus analysis above, we group candidate phrases into the **14 required rule
categories** (8 positive, 6 negative) and report their raw (un-guarded) frequency in the corpus.
These raw counts are the *starting point*; Section 7 adds negation-lookbehind guards where Section 5
showed the phrase can be reversed by a preceding negation word, and reports the **final, guarded**
coverage."""))

cells.append(new_code_cell(r'''RULE_CANDIDATES = {
    # --- positive (8) -----------------------------------------------
    'revisit_intention':      ['sẽ quay lại', 'sẽ ghé lại', 'quay lại lần sau', 'chắc chắn sẽ quay lại'],
    'repeat_purchase':         ['ăn lại', 'mua lại', 'order lại', 'đặt lại'],
    'recommendation':          ['recommend', 'giới thiệu cho bạn bè', 'nên thử', 'đáng thử'],
    'value_for_money':         ['đáng đồng tiền', 'xứng đáng số tiền', 'đáng giá số tiền'],
    'strong_satisfaction':      ['hài lòng', 'rất hài lòng', 'tuyệt vời', 'xuất sắc', '10 điểm'],
    'no_complaint':             ['không có gì để chê', 'chẳng có gì phàn nàn'],
    'loyalty':                  ['khách quen', 'ăn ở đây nhiều', 'ghé thường xuyên'],
    'advocacy':                 ['ủng hộ', 'support quán', 'pr cho quán'],
    # --- negative (6) -------------------------------------------------
    'no_revisit':               ['không quay lại', 'không bao giờ quay lại', 'đừng quay lại'],
    'strong_dissatisfaction':   ['rất thất vọng', 'quá tệ', 'kinh khủng', 'không hài lòng', '0 điểm'],
    'not_worth_it':             ['không xứng đáng', 'không đáng tiền', 'phí tiền'],
    'waiting_problem':          ['chờ lâu', 'đợi lâu', 'phục vụ chậm'],
    'frustration':              ['bực mình', 'khó chịu', 'tức giận'],
    'regret':                   ['hối hận', 'tiếc tiền', 'lãng phí tiền'],
}

print(f"Rule candidate categories: {len(RULE_CANDIDATES)} (expected 14)")
assert len(RULE_CANDIDATES) == 14

print(f"\n{'category':25s} {'raw phrase hits (any candidate phrase)':>40s}")
for cat, phrases in RULE_CANDIDATES.items():
    n = sum(1 for t in corpus if any(re.search(re.escape(p), t) for p in phrases))
    print(f"{cat:25s} {n:40d}")
'''))

# =====================================================================
# Section 7: Create and Validate rules.json
# =====================================================================
cells.append(new_markdown_cell(r"""## Section 7 — Create and Validate `overall_satisfaction_rules.json`

This section defines the **final** 14 rule categories as a Python dict, refined from the candidates
above using the negation findings of Section 5 (e.g. guarding `thất vọng`, `xuất sắc`, `tệ`,
`recommend`, `ủng hộ`, `quay lại`, `xứng đáng` against the reversal idioms discovered there, and
adding a dedicated "không hài lòng" pattern to `strong_dissatisfaction`).

Each rule has the schema `{"score": float, "description": str, "patterns": [regex, ...]}`.
The dict is **validated** (14 categories, 8 positive / 6 negative, every pattern compiles as a valid
Python regex) and then **written** to `overall_satisfaction_rules.json`. The rule engine
(Section 9) loads this file from disk — the patterns are *data*, not hardcoded engine logic."""))

cells.append(new_code_cell(r'''RULES_DICT = {
    "revisit_intention": {
        "score": 0.5,
        "description": "Reviewer explicitly expresses intention to return or visit the restaurant again in the future. This is a global satisfaction signal independent of any single aspect (food/service/atmosphere/price). The '(quay lai|tro lai|ghe lai) lan (sau|toi|nua)' pattern additionally carries '(khong|chang|ko) bao gio' guards because corpus analysis found '(khong|ko) bao gio quay lai lan nua' (=will never come back again) is a common NEGATIVE idiom that would otherwise be misclassified as a revisit intention. The 'se (quay lai|...)' pattern carries '(khong|ko) nghi' guards because corpus analysis found '(khong|ko) nghi se quay lai' (=don't think [I'll] come back) is a NEGATIVE idiom that would otherwise be misclassified as a revisit intention.",
        "patterns": [
            r"(?<!không )(?<!chẳng )(?<!chưa )(?<!ko )(?<!không nghĩ )(?<!ko nghĩ )sẽ (quay lại|trở lại|ghé lại|ghé|đến lại)",
            r"(?<!không )(?<!chẳng )(?<!chưa )(?<!ko )(?<!không bao giờ )(?<!chẳng bao giờ )(?<!ko bao giờ )(quay lại|trở lại|ghé lại) lần (sau|tới|nữa)",
            r"lần sau (sẽ |chắc |nhất định )?(quay lại|ghé|trở lại)",
            r"nhất định (sẽ )?(quay lại|ghé lại|trở lại)",
            r"chắc chắn (sẽ )?(quay lại|ghé lại|trở lại)",
        ],
    },
    "repeat_purchase": {
        "score": 0.4,
        "description": "Reviewer indicates they will order, buy, or eat the same item again (repeat-purchase intention for the product/dish, distinct from revisiting the venue itself). All patterns carry '(khong|chang|ko) bao gio' guards because corpus analysis found '(khong|chac chan khong|ko) bao gio an/mua/dat lai' (=will never order/buy/eat again) are common NEGATIVE idioms that would otherwise be misclassified as repeat-purchase intention.",
        "patterns": [
            r"(?<!không )(?<!chẳng )(?<!chưa )(?<!ko )(?<!không bao giờ )(?<!chẳng bao giờ )(?<!ko bao giờ )ăn lại",
            r"(?<!không )(?<!chẳng )(?<!chưa )(?<!ko )(?<!không bao giờ )(?<!chẳng bao giờ )(?<!ko bao giờ )mua lại",
            r"(?<!không )(?<!chẳng )(?<!chưa )(?<!ko )(?<!không bao giờ )(?<!chẳng bao giờ )(?<!ko bao giờ )order lại",
            r"(?<!không )(?<!chẳng )(?<!chưa )(?<!ko )(?<!không bao giờ )(?<!chẳng bao giờ )(?<!ko bao giờ )đặt lại",
            r"(?<!không )(?<!chẳng )(?<!chưa )(?<!ko )(?<!không bao giờ )(?<!chẳng bao giờ )(?<!ko bao giờ )gọi lại (món|đồ)",
        ],
    },
    "recommendation": {
        "score": 0.4,
        "description": "Reviewer recommends the restaurant or its food/service to other people - a word-of-mouth endorsement of the overall experience.",
        "patterns": [
            r"(?<!không )(?<!chẳng )(?<!chưa )(?<!ko )(?<!sẽ không )recommend",
            r"(?<!không )(?<!chẳng )(?<!chưa )giới thiệu (cho |với )?(bạn bè|mọi người|gia đình|người thân)",
            r"(?<!không )(?<!chẳng )(?<!chưa )(?<!ko )nên (thử|ăn thử|ghé|đến|quay lại)",
            r"(?<!không )(?<!chẳng )(?<!chưa )(?<!ko )đáng (để )?thử",
            r"(?<!không )(?<!chẳng )(?<!chưa )(?<!ko )rất đáng (thử|ghé)",
        ],
    },
    "value_for_money": {
        "score": 0.3,
        "description": "Reviewer explicitly states the price paid is worth the overall value/quality received. This is a global price-value judgment, distinct from aspect-specific price comments such as 'gia re' or 'gia hop ly' which are excluded from this rule set.",
        "patterns": [
            r"(?<!không )(?<!chẳng )(?<!chưa )đáng đồng tiền",
            r"(?<!không )(?<!chẳng )(?<!chưa )xứng đáng (với )?(số tiền|đồng tiền|tiền bỏ ra)",
            r"(?<!không )(?<!chẳng )(?<!chưa )đáng giá (với )?(số tiền|đồng tiền|tiền bỏ ra)",
        ],
    },
    "strong_satisfaction": {
        "score": 0.6,
        "description": "Reviewer expresses strong overall satisfaction or delight with the experience as a whole, not tied to a single aspect (food/service/atmosphere/price). 'xuat sac'/'hoan hao' patterns carry extra lookbehind guards for hedged-negation idioms such as 'khong qua xuat sac' (=not too excellent) discovered during corpus analysis. The bare 'hai long' pattern additionally carries '(?<!ko )', '(?<!khong duoc )', '(?<!ko duoc )', '(?<!khong thay )' and '(?<!chua lam )' guards because corpus analysis found 'ko hai long' (=not satisfied) and '(khong|ko) duoc hai long' (=wasn't satisfied) are common NEGATIVE idioms that would otherwise be misclassified as strong satisfaction. Every 'khong X' guard on 'xuat sac' is mirrored by a 'ko X' guard (and 'tuyet voi' gains a 'khong/ko ngon' guard) because 'ko' is the same informal abbreviation of 'khong' found during corpus analysis of the 'hai long' pattern, e.g. 'ko qua xuat sac' (=not too excellent) and 'ko ngon tuyet voi' (=not wonderfully delicious) are common hedged-negation idioms.",
        "patterns": [
            r"(?<!không )(?<!chẳng )(?<!chưa )rất hài lòng",
            r"cực kỳ hài lòng",
            r"vô cùng hài lòng",
            r"(?<!không )(?<!chẳng )(?<!chưa )(?<!ko )(?<!không được )(?<!ko được )(?<!không thấy )(?<!chưa làm )hài lòng",
            r"(?<!không )(?<!chẳng )(?<!chưa )(?<!không ngon )(?<!ko ngon )tuyệt vời",
            r"(?<!không )(?<!chẳng )(?<!chưa )(?<!ko )(?<!không quá )(?<!ko quá )(?<!không phải )(?<!ko phải )(?<!không phải quá )(?<!ko phải quá )(?<!không hẳn )(?<!ko hẳn )(?<!không gọi là )(?<!ko gọi là )(?<!không có gì )(?<!ko có gì )(?<!không có gì quá )(?<!ko có gì quá )(?<!không ngon )(?<!ko ngon )(?<!không phải ngon )(?<!ko phải ngon )xuất sắc",
            r"(?<!không )(?<!chẳng )(?<!chưa )(?<!không đến mức )(?<!không gọi là )hoàn hảo",
            r"\b10/10\b|\b10 điểm\b",
        ],
    },
    "no_complaint": {
        "score": 0.3,
        "description": "Reviewer explicitly states they have no complaints or issues with the overall experience. The leading 'khong'/'chang'/'ko' here is part of the positive idiom itself, so it is intentionally not guarded against; the first pattern's '(khong|ko)' alternation adds the same 'ko'-abbreviation coverage used throughout this rule set (e.g. 'ko co gi de che', 19 additional reviews). A trailing word boundary on 'che' in all three patterns fixes a pre-existing bug where 'che' matched as a substring-prefix of the unrelated word 'chenh' (=differ/vary), e.g. 'khong chenh lech nhieu' (=doesn't vary much, a price-consistency remark unrelated to complaints) was previously misclassified as no_complaint (3 reviews).",
        "patterns": [
            r"(không|ko) (có )?(gì )?(để |phải )?(chê\b|phàn nàn|than phiền)",
            r"chẳng (có )?(gì )?(để |phải )?(chê\b|phàn nàn|than phiền)",
            r"không (một|1) (lời|điều|chỗ) (chê\b|phàn nàn)",
        ],
    },
    "loyalty": {
        "score": 0.5,
        "description": "Reviewer identifies as a regular or loyal customer / long-term patron of the restaurant.",
        "patterns": [
            r"khách quen",
            r"ăn (ở đây|quán này|chỗ này) (nhiều|hoài|suốt|thường xuyên|miết)",
            r"(?<!không )(?<!chẳng )(?<!chưa )(?<!ko )ghé (đây|quán này|chỗ này) (thường xuyên|hoài|suốt|nhiều lần)",
            r"lần thứ (\d+|mấy|bao nhiêu|n) (ăn|ghé|đến|quay lại)",
            r"(?<!không )(?<!chẳng )(?<!chưa )trung thành",
        ],
    },
    "advocacy": {
        "score": 0.5,
        "description": "Reviewer actively promotes or endorses the restaurant to others, going beyond a simple recommendation (active word-of-mouth advocacy / support for the business). The 'ung ho' pattern requires a leading word boundary because corpus analysis found it otherwise false-matches as a substring of unrelated words such as 'lung hop' (=the box leaks/breaks, about packaging); it also carries '(khong|ko) muon' guards because '(khong|ko) muon ung ho' (=don't want to support) is a NEGATIVE idiom that would otherwise be misclassified as advocacy.",
        "patterns": [
            r"\b(?<!không )(?<!chẳng )(?<!chưa )(?<!ko )(?<!sẽ không )(?<!không muốn )(?<!ko muốn )ủng hộ",
            r"(?<!không )(?<!chẳng )(?<!chưa )(?<!ko )support quán",
            r"(?<!không )(?<!chẳng )(?<!chưa )pr cho quán",
            r"(?<!không )(?<!chẳng )(?<!chưa )quảng cáo không công",
        ],
    },
    "no_revisit": {
        "score": -0.5,
        "description": "Reviewer explicitly states they will not return to the restaurant in the future. The leading '(khong|ko)' alternation in the first and fourth patterns mirrors the 'ko'-abbreviation handling used throughout this rule set (see strong_satisfaction): corpus analysis found 'ko bao gio quay lai' / 'ko nen an o day' are common informal spellings of 'khong bao gio quay lai' / 'khong nen an o day' that the bare 'khong'-only patterns missed (76 additional reviews corpus-wide).",
        "patterns": [
            r"(sẽ )?(không|ko) (bao giờ )?(quay lại|trở lại|ghé lại|đến lại|ăn lại)( nữa)?",
            r"chắc (chắn )?không (quay lại|trở lại|ghé|ghé lại)",
            r"đừng (quay lại|ghé|trở lại)",
            r"(không|ko) nên (đến|ghé|ăn) (ở đây|quán này|chỗ này)",
            r"(một|1) lần (rồi thôi|là đủ|cho biết|và không bao giờ trở lại)",
        ],
    },
    "strong_dissatisfaction": {
        "score": -0.6,
        "description": "Reviewer expresses strong overall dissatisfaction or disappointment with the experience as a whole, not tied to a single aspect. The 'that vong' and 'te' patterns carry extended lookbehind guards for common reversal idioms ('se khong lam ban that vong' = won't disappoint you; 'khong qua te' = not too bad) discovered during corpus analysis. A dedicated '(khong|ko) hai long' (=not satisfied) pattern is included because it is a direct negative-satisfaction statement, not a hedge of a positive word; the '(khong|ko)' alternation is the same 'ko'-abbreviation found during corpus analysis of strong_satisfaction's 'hai long' guard ('ko hai long' is the exact idiom that pattern guards AGAINST as a positive - here it is the trigger FOR strong_dissatisfaction, adding 14 reviews that previously triggered no rule at all).",
        "patterns": [
            r"(?<!không )(?<!chẳng )(?<!chưa )(?<!không tới nỗi )(?<!không đến nổi )(?<!không đến nỗi )rất (thất vọng|tệ)",
            r"cực kỳ (thất vọng|tệ)",
            r"(?<!không )(?<!chẳng )(?<!chưa )(?<!không tới nỗi )(?<!không đến nổi )(?<!không đến nỗi )quá (tệ|thất vọng)",
            r"(?<!không )(?<!chẳng )(?<!chưa )(?<!không hề )(?<!không bị )(?<!không phải )(?<!không làm bạn )(?<!không làm mình )(?<!không khiến mình )(?<!không làm các bạn )(?<!không làm mọi người )(?<!không bao giờ làm mình )(?<!không cảm thấy chút nào )thất vọng",
            r"tệ (nhất|kinh khủng|không thể tả)",
            r"kinh khủng",
            r"(không|ko) (thấy |cảm thấy |được )?hài lòng",
            r"\b0/10\b|\b1/10\b|\b0 điểm\b",
        ],
    },
    "not_worth_it": {
        "score": -0.5,
        "description": "Reviewer states the price paid was not worth it relative to the overall quality received - a global price-value judgment (negative counterpart of value_for_money). The second pattern's '(khong|ko)' alternation adds the same 'ko'-abbreviation coverage used throughout this rule set (e.g. 'ko dang dong tien', 1 additional review).",
        "patterns": [
            r"không xứng (đáng )?(với )?(số tiền|đồng tiền|tiền bỏ ra)",
            r"(không|ko) đáng (giá|với)? ?(số tiền|đồng tiền|tiền bỏ ra)",
            r"chẳng đáng (số )?tiền",
            r"phí tiền",
            r"tiền mất tật mang",
        ],
    },
    "waiting_problem": {
        "score": -0.4,
        "description": "Reviewer reports significant waiting-time issues, indicating slow overall service delivery (not a specific food/service quality complaint, but an overall experience friction). The 'cho/doi (rat|qua)? lau' patterns carry '(khong|ko) (phai|bi|can|de)?', 'chua (phai)?' and 'dung' guards because corpus analysis found '(khong|ko) phai doi/cho (qua)? lau' (=don't have to wait long) is a common POSITIVE idiom that would otherwise be misclassified as a waiting-time problem.",
        "patterns": [
            r"(?<!không )(?<!không phải )(?<!không bị )(?<!không cần )(?<!không để )(?<!ko )(?<!ko phải )(?<!ko bị )(?<!ko cần )(?<!ko để )(?<!chẳng )(?<!chưa )(?<!chưa phải )(?<!đừng )chờ (rất |quá )?lâu",
            r"(?<!không )(?<!không phải )(?<!không bị )(?<!không cần )(?<!không để )(?<!ko )(?<!ko phải )(?<!ko bị )(?<!ko cần )(?<!ko để )(?<!chẳng )(?<!chưa )(?<!chưa phải )(?<!đừng )đợi (rất |quá )?lâu",
            r"chờ (\d+) (phút|tiếng|giờ)",
            r"đợi (\d+) (phút|tiếng|giờ)",
            r"phục vụ (quá |rất )?chậm",
            r"lâu (kinh khủng|ơi là lâu|không chịu được)",
        ],
    },
    "frustration": {
        "score": -0.4,
        "description": "Reviewer expresses frustration, anger, or annoyance with the overall experience. The 'kho chiu' pattern carries '(khong|chang|chua|ko)', '(khong|ko) he' and 'khong bi/thay' guards because corpus analysis found '(khong|ko) (he) kho chiu' (=not at all annoying) is a common POSITIVE idiom that would otherwise be misclassified as frustration.",
        "patterns": [
            r"bực (mình|bội)",
            r"(?<!không )(?<!chẳng )(?<!chưa )(?<!ko )(?<!không hề )(?<!ko hề )(?<!không bị )(?<!không thấy )khó chịu",
            r"(?<!không )(?<!chẳng )(?<!chưa )nổi (giận|cáu)",
            r"tức (giận|mình)",
            r"thái độ (kém|tệ|khó chịu)",
        ],
    },
    "regret": {
        "score": -0.4,
        "description": "Reviewer expresses regret about having visited or spent money/time at the restaurant. 'hoi han' and 'uong cong/tien' carry negation guards because corpus analysis found 'khong uong cong' (=not a waste of effort, i.e. well worth it) and 'khong (chut) hoi han' (=no regrets) are common POSITIVE idioms that would otherwise be misclassified as regret.",
        "patterns": [
            r"(?<!không )(?<!chẳng )(?<!chưa )(?<!ko )(?<!không chút )(?<!chẳng chút )hối hận",
            r"(?<!không )(?<!chẳng )(?<!chưa )(?<!ko )tiếc (tiền|công|thời gian)",
            r"(?<!không )(?<!chẳng )(?<!chưa )(?<!ko )(uổng|lãng phí) (tiền|công|thời gian)",
            r"biết (vậy|thế) (thì |là )?(đã )?không (đến|ghé|ăn)",
        ],
    },
}

print(f"Total rule categories: {len(RULES_DICT)}")
'''))

cells.append(new_markdown_cell(r"""### 7.1 Schema and regex validation

Before saving, we check:
1. Exactly 14 categories, with the exact 8 positive / 6 negative names required by the spec.
2. Every rule has the `{score, description, patterns}` schema with the correct types.
3. Every pattern compiles as a valid Python regex (this also catches malformed/variable-width
   lookbehind assertions, which Python's `re` module rejects at compile time)."""))

cells.append(new_code_cell(r'''REQUIRED_POSITIVE = {
    'revisit_intention', 'repeat_purchase', 'recommendation', 'value_for_money',
    'strong_satisfaction', 'no_complaint', 'loyalty', 'advocacy',
}
REQUIRED_NEGATIVE = {
    'no_revisit', 'strong_dissatisfaction', 'not_worth_it', 'waiting_problem',
    'frustration', 'regret',
}
REQUIRED_ALL = REQUIRED_POSITIVE | REQUIRED_NEGATIVE

assert set(RULES_DICT.keys()) == REQUIRED_ALL, \
    f"Rule set mismatch. Missing: {REQUIRED_ALL - set(RULES_DICT)}, Extra: {set(RULES_DICT) - REQUIRED_ALL}"
assert len(RULES_DICT) == 14

n_compiled = 0
for name, rule in RULES_DICT.items():
    assert set(rule.keys()) == {'score', 'description', 'patterns'}, f"{name}: bad schema keys {rule.keys()}"
    assert isinstance(rule['score'], (int, float)), f"{name}: score must be numeric"
    assert isinstance(rule['description'], str) and len(rule['description']) > 0, f"{name}: description must be non-empty string"
    assert isinstance(rule['patterns'], list) and len(rule['patterns']) > 0, f"{name}: patterns must be a non-empty list"

    expected_sign = 1 if name in REQUIRED_POSITIVE else -1
    actual_sign = 1 if rule['score'] > 0 else -1
    assert actual_sign == expected_sign, f"{name}: score sign {rule['score']} does not match category polarity"

    for pattern in rule['patterns']:
        re.compile(pattern)  # raises re.error if invalid (incl. variable-width lookbehind)
        n_compiled += 1

print(f"Validated {len(RULES_DICT)} rule categories "
      f"({len(REQUIRED_POSITIVE)} positive, {len(REQUIRED_NEGATIVE)} negative)")
print(f"Compiled {n_compiled} regex patterns successfully")
'''))

cells.append(new_markdown_cell(r"""### 7.2 Save `overall_satisfaction_rules.json`"""))

cells.append(new_code_cell(r'''rules_json = {
    "metadata": {
        "version": "1.0",
        "generated_at": RUN_TIMESTAMP,
        "scale_min": 0,
        "scale_max": 10,
        "total_categories": len(RULES_DICT),
        "positive_categories": len(REQUIRED_POSITIVE),
        "negative_categories": len(REQUIRED_NEGATIVE),
        "language": "vi",
        "notes": ("Patterns are matched against normalized review text (Unicode NFC, lowercased, "
                  "whitespace-collapsed) using Python 're' with re.search. Negative lookbehind "
                  "assertions guard against negated forms ('khong'/'chang'/'chua'/'ko'/'se khong' "
                  "and several multi-word reversal idioms) that would reverse a pattern's polarity. "
                  "Each rule's 'score' is added to avg_rating (0-10 scale) for every matched pattern "
                  "occurrence in a review."),
    },
    "rules": RULES_DICT,
}

with open(RULES_PATH, 'w', encoding='utf-8') as f:
    json.dump(rules_json, f, ensure_ascii=False, indent=2)

print(f"Wrote {RULES_PATH} ({RULES_PATH.stat().st_size:,} bytes)")

# Round-trip check
with open(RULES_PATH, encoding='utf-8') as f:
    reloaded = json.load(f)
assert reloaded == rules_json
print("Round-trip JSON load matches in-memory dict.")
'''))

# =====================================================================
# Section 8: Text Normalization
# =====================================================================
cells.append(new_markdown_cell(r"""## Section 8 — Text Normalization

`normalize_text()` is the single normalization function used by the rule engine. It is applied to
`comment_clean` before pattern matching so that regex patterns (written in lowercase) match
consistently regardless of capitalization, accent composition, or extra whitespace.

Steps:
1. Unicode **NFC normalization** (composes combining diacritics into single code points — Vietnamese
   text scraped from the web can mix NFC and NFD forms for the same visible character).
2. **Lowercasing**.
3. **Whitespace collapsing** (multiple spaces/newlines/tabs -> single space, trim ends)."""))

cells.append(new_code_cell(r'''def normalize_text(text: str) -> str:
    """Normalize Vietnamese review text for rule-pattern matching.

    Applies Unicode NFC normalization, lowercasing, and whitespace collapsing.
    Non-string input (e.g. NaN) returns an empty string.

    Args:
        text: Raw review text (e.g. from the 'comment_clean' column).

    Returns:
        Normalized text: NFC-composed, lowercase, single-spaced, trimmed.
    """
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize('NFC', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# --- Unit tests ---------------------------------------------------------
assert normalize_text(None) == ""
assert normalize_text(np.nan) == ""
assert normalize_text("  Quán   NGON\n\tlắm!  ") == "quán ngon lắm!"
assert normalize_text("Đồ ăn") == normalize_text(unicodedata.normalize('NFD', "Đồ ăn")), \
    "NFC and NFD encodings of the same text must normalize identically"

print("normalize_text() unit tests passed.")
print(repr(normalize_text("  Quán   NGON\n\tlắm! Sẽ QUAY LẠI lần sau.  ")))
'''))

# =====================================================================
# Section 9: Rule Engine
# =====================================================================
cells.append(new_markdown_cell(r"""## Section 9 — Rule Engine

The rule engine is **fully data-driven**: `RULES` is loaded from `overall_satisfaction_rules.json`
on disk (not the in-memory `RULES_DICT` from Section 7), and every function below operates purely
on that loaded structure plus the normalized review text. This guarantees that re-running this
notebook against an edited `overall_satisfaction_rules.json` (without touching the engine code)
changes the output — i.e. rules are configuration, not code.

Four functions:

- `find_matching_rules(text, rules)` — for normalized `text`, return every `(rule_name, pattern,
  matched_span)` triple where `pattern` (one of `rules[rule_name]['patterns']`) matches.
- `calculate_adjustment(matches, rules)` — sum `rules[rule_name]['score']` over all matches
  (`overall_adjustment`).
- `generate_evidence(matches, rules)` — turn matches into a structured, JSON-serializable list of
  evidence records (rule name, score, matched text, category polarity).
- `apply_rules(text, rules)` — convenience wrapper combining the three above for one review."""))

cells.append(new_code_cell(r'''with open(RULES_PATH, encoding='utf-8') as f:
    RULES_FILE = json.load(f)
RULES = RULES_FILE['rules']

# Pre-compile every pattern once (keyed by rule name) for performance.
_COMPILED_PATTERNS = {
    name: [re.compile(p) for p in rule['patterns']]
    for name, rule in RULES.items()
}

print(f"Loaded {len(RULES)} rule categories from {RULES_PATH}")
'''))

cells.append(new_code_cell(r'''def find_matching_rules(text: str, rules: dict = RULES) -> list[tuple[str, str, str]]:
    """Find every rule pattern that matches the given normalized text.

    Args:
        text: Normalized review text (output of normalize_text()).
        rules: Mapping of rule_name -> {"score", "description", "patterns"}.

    Returns:
        A list of (rule_name, pattern, matched_text) for every pattern that matches
        at least once. If a rule has multiple patterns matching, each contributes its
        own tuple (so a rule can fire more than once per review).
    """
    matches = []
    for rule_name, rule in rules.items():
        for pattern in rule['patterns']:
            compiled = re.compile(pattern)
            m = compiled.search(text)
            if m:
                matches.append((rule_name, pattern, m.group(0)))
    return matches


def calculate_adjustment(matches: list[tuple[str, str, str]], rules: dict = RULES) -> float:
    """Sum the rule scores for every DISTINCT rule category that matched.

    Each rule category contributes its score at most once per review, even if
    several of its patterns match (e.g. "sẽ quay lại" and "chắc chắn sẽ quay lại"
    are two patterns of 'revisit_intention' that can both match the same sentence -
    this must not double-count the same underlying signal).

    Args:
        matches: Output of find_matching_rules().
        rules: Same rules mapping used to produce `matches`.

    Returns:
        The total overall_adjustment: sum of rules[rule_name]['score'] over the
        set of unique rule_names present in `matches`.
    """
    unique_rules = dict.fromkeys(rule_name for rule_name, _pattern, _text in matches)
    return sum(rules[rule_name]['score'] for rule_name in unique_rules)


def generate_evidence(matches: list[tuple[str, str, str]], rules: dict = RULES) -> list[dict]:
    """Build a structured, JSON-serializable evidence list from rule matches.

    One evidence record is produced per DISTINCT rule category (using its first
    matching pattern/span), matching the deduplication done in calculate_adjustment()
    so that sum(e['score'] for e in evidence) == calculate_adjustment(matches, rules).

    Args:
        matches: Output of find_matching_rules().
        rules: Same rules mapping used to produce `matches`.

    Returns:
        A list of dicts, one per distinct triggered rule, each with:
          - "rule": rule category name
          - "score": the score contribution of this rule
          - "polarity": "positive" or "negative" (sign of the rule's score)
          - "matched_text": the exact substring that matched (first matching pattern)
          - "pattern": the regex pattern that matched
    """
    seen = set()
    evidence = []
    for rule_name, pattern, matched_text in matches:
        if rule_name in seen:
            continue
        seen.add(rule_name)
        score = rules[rule_name]['score']
        evidence.append({
            "rule": rule_name,
            "score": score,
            "polarity": "positive" if score > 0 else "negative",
            "matched_text": matched_text,
            "pattern": pattern,
        })
    return evidence


def apply_rules(text: str, rules: dict = RULES) -> tuple[float, list[str], list[dict]]:
    """Apply the full rule engine to one normalized review text.

    Args:
        text: Normalized review text (output of normalize_text()).
        rules: Mapping of rule_name -> {"score", "description", "patterns"}.

    Returns:
        A 3-tuple:
          - overall_adjustment (float): sum of matched rule scores.
          - rules_triggered (list[str]): unique rule names that matched at least once,
            in the order they were first matched.
          - evidence (list[dict]): structured evidence records (see generate_evidence()).
    """
    matches = find_matching_rules(text, rules)
    adjustment = calculate_adjustment(matches, rules)
    evidence = generate_evidence(matches, rules)
    triggered = list(dict.fromkeys(rule_name for rule_name, _, _ in matches))
    return adjustment, triggered, evidence
'''))

cells.append(new_markdown_cell(r"""### 9.1 Rule engine unit tests

A handful of hand-crafted sentences exercise the positive case, the negative case, the
negation-guard case, and the no-match case."""))

cells.append(new_code_cell(r'''test_cases = [
    ("Món ăn ngon, chắc chắn sẽ quay lại lần sau!", "revisit_intention", True),
    ("Sẽ không bao giờ quay lại nữa, quá tệ.", "no_revisit", True),
    ("Sẽ không bao giờ quay lại nữa, quá tệ.", "strong_dissatisfaction", True),
    ("Sẽ không bao giờ quay lại nữa, quá tệ.", "revisit_intention", False),
    ("Phục vụ thái độ như vậy, ko bao giờ quay lại.", "no_revisit", True),
    ("Thái độ nhân viên thì mình ko hài lòng lắm.", "strong_dissatisfaction", True),
    ("Thái độ nhân viên thì mình ko hài lòng lắm.", "strong_satisfaction", False),
    ("Quán rất ổn, ko có gì để chê cả.", "no_complaint", True),
    ("Giá cả các quán ở đây không chênh lệch nhiều.", "no_complaint", False),
    ("Quán này tạm được, không có gì đặc biệt.", None, None),  # no rule should fire
    ("Đồ ăn ngon nhưng phục vụ thì quá chậm, chờ rất lâu.", "waiting_problem", True),
    ("Không hề thất vọng, sẽ giới thiệu cho bạn bè!", "strong_dissatisfaction", False),
    ("Không hề thất vọng, sẽ giới thiệu cho bạn bè!", "recommendation", True),
]

for text, expected_rule, expected_present in test_cases:
    norm = normalize_text(text)
    _, triggered, _ = apply_rules(norm)
    if expected_rule is None:
        assert triggered == [], f"Expected NO rules for {text!r}, got {triggered}"
    else:
        present = expected_rule in triggered
        assert present == expected_present, (
            f"Text {text!r}: expected rule '{expected_rule}' present={expected_present}, "
            f"got triggered={triggered}"
        )

print(f"All {len(test_cases)} rule-engine unit tests passed.")

# Show a worked example
sample_text = "Đồ ăn ngon nhưng phục vụ thì quá chậm, chờ rất lâu. Chắc chắn sẽ quay lại!"
norm = normalize_text(sample_text)
adj, triggered, evidence = apply_rules(norm)
print(f"\nExample review: {sample_text!r}")
print(f"overall_adjustment = {adj:+.2f}")
print(f"rules_triggered    = {triggered}")
for e in evidence:
    print(f"  {e}")
'''))

with open('notebook/01_generate_overall_satisfaction.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"Notebook now has {len(cells)} cells")
