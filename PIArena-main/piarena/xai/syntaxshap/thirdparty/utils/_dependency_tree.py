from collections import deque
import logging
import pandas as pd
import spacy

log = logging.getLogger(__name__)

# NOTE (PIArena vendoring): dropped `import textdescriptives as td` and the
# `compute_dependency_distance` function that used it — neither is called by
# `get_token_dependency_tree`, and textdescriptives pulls in a heavy dependency
# chain (benepar, pyphen, ftfy, ...) we don't need. See piarena/xai plan for context.

class TreeNode:
    def __init__(self, data):
        """Initialize a TreeNode with data, parent, and children."""
        self.data = data
        self.parent = None
        self.children = []

    def add_child(self, child):
        """Add a child TreeNode to this node."""
        child.parent = self
        self.children.append(child)

def spacy_doc_to_tree(doc):
    """
    Convert a SpaCy Doc object into a tree structure represented by TreeNodes.

    Args:
        doc (spacy.tokens.Doc): SpaCy document object.

    Returns:
        TreeNode: The root node of the constructed tree.
    """
    nodes = [TreeNode(token) for token in doc]

    # Create the tree structure by connecting parent-child relationships
    for token, node in zip(doc, nodes):
        if token.head.i == token.i:  # Skip root node (head is itself)
            continue
        parent_node = nodes[token.head.i]
        parent_node.add_child(node)

    # Find and return the root node
    root = next(node for node in nodes if node.parent is None)
    return root

def create_dataframe_from_tree(root):
    """
    Create a pandas DataFrame from a tree structure.

    Args:
        root (TreeNode): Root node of the tree.

    Returns:
        pd.DataFrame: DataFrame representing the tree structure.
    """
    if not root:
        return pd.DataFrame()

    # Data list to hold information about each node
    data = []
    queue = deque([(root, 0, None)])

    while queue:
        node, level, parent = queue.popleft()

        # Gather node data
        node_data = {
            "word": node.data.text,
            "word_position": node.data.i,
            "level": level,
            "level_weight": 1 / (level + 1),
            "parent": parent.data.text if parent else None
        }
        data.append(node_data)

        # Add children to the queue for processing
        for child in node.children:
            queue.append((child, level + 1, node))

    # Create DataFrame from collected data
    df = pd.DataFrame(data)
    df = df.sort_values(by='level')
    return df

_NLP_CACHE = {}


def _load_nlp(model_name="en_core_web_sm"):
    """Cached `spacy.load` — the original code called `spacy.load(...)` fresh
    inside `spacy_dependency_tree` on *every* call, i.e. once per sentence in
    `ClassifierSyntaxExplainer.explain_context`'s per-sentence loop. Reloading
    a spaCy pipeline from disk is expensive and the model never changes
    between calls within one process, so this caches it after the first load.
    """
    if model_name not in _NLP_CACHE:
        _NLP_CACHE[model_name] = spacy.load(model_name)
    return _NLP_CACHE[model_name]


def spacy_parse_words_and_tree(sentence, nlp=None):
    """
    Parse `sentence` with spaCy ONCE, returning both its own token strings
    (in doc order, real words only) and the dependency tree DataFrame built
    from that SAME doc — the two are guaranteed to share an identical word
    segmentation and position indexing by construction.

    PIArena fix (empty/near-empty tree bug): the previous version had
    `spacy_dependency_tree` parse `sentence` with spaCy to build the tree,
    while `compute_position_mapping` *independently* re-split the same
    `sentence` with a hand-rolled regex (`\\b.*?\\S.*?(?:\\b|$)`) to build the
    word list used for the tokenizer<->word alignment. `get_token_dependency_tree`
    then joined the two DataFrames on `['word', 'word_position']` — which
    silently requires both word *text* and word *position index* to agree
    between the two independently-derived segmentations. They very often
    don't: the regex splits every punctuation character into its own "word"
    (`"Don't"` -> `["Don", "'", "t"]`, `"U.S."` -> `["U", ".", "S", "."]`),
    while spaCy's tokenizer keeps contractions/abbreviations/decimals
    together (`"Don't"` -> `["Do", "n't"]`, etc.). The first such divergence
    in a sentence shifts every later word's index out of alignment between
    the two DataFrames, so the inner join drops it and everything after it —
    on real prose (squad_v2 Wikipedia-style text is full of contractions,
    decimals, hyphenated words, quoted phrases), a single early divergence
    was enough to leave a real, non-trivial sentence with an empty or
    near-empty tree, silently zeroing out its SHAP contribution instead of
    actually explaining it. Fixed by never deriving two separate word lists
    for the same sentence in the first place — both the word list and the
    tree now come from the one `doc` this function builds.

    Returns
    -------
    (words, tree_df) : (list[str], pd.DataFrame)
        `words[i]` is real word `i` of `sentence` (0-based, spaCy's own
        tokenization); `tree_df['word_position']` indexes into that same
        list — by construction, never independently re-derived.
    """
    nlp = nlp or _load_nlp()
    # A sentinel token is appended so spaCy always has something to attach
    # trailing punctuation/dependency edges to; dropped by *position* below
    # (the last token of the doc), not by matching the text "MASK" — matching
    # by text would wrongly drop a real word if the sentence being explained
    # legitimately contains the word "MASK" (rare, but that's exactly the
    # kind of position/identity confusion this fix is about).
    doc = nlp(sentence + ' MASK')
    real_tokens = list(doc)[:-1]
    words = [t.text for t in real_tokens]

    tree_root = spacy_doc_to_tree(doc)
    tree_df = create_dataframe_from_tree(tree_root)
    tree_df = tree_df[tree_df['word_position'] < len(words)]

    return words, tree_df


def compute_position_mapping(words, sentence, tokenizer):
    """
    Compute the mapping between words and subtokens in a sentence.

    `words` must be the word segmentation to align the tokenizer's subtokens
    against — pass the list `spacy_parse_words_and_tree` returns so it's the
    *same* segmentation the dependency tree's `word_position` column indexes
    into (see that function's docstring for why using two different
    segmentations here used to silently produce empty/near-empty trees).
    """
    df_words = pd.DataFrame({'word': words, 'word_position': range(len(words))})

    token_ids = tokenizer(sentence)['input_ids']
    df_tokens = pd.DataFrame({'token_id': token_ids, 'token_position': range(len(token_ids))})
    df_tokens['token'] = [tokenizer.decode([token_id]) for token_id in df_tokens['token_id']]

    pos_token_to_word = {}
    k = 0

    for i in range(len(words)):
        word = words[i]
        word_len = 0
        while word_len < len(word):
            if k >= len(token_ids):
                # PIArena fix #3: the loop below assumes every subtoken's
                # decoded (space-stripped) form contributes >=1 char towards
                # `word`'s length budget — true often enough, but not when a
                # subtoken decodes to whitespace only (e.g. around a double
                # space — PIArena's own `inject()` can produce these when
                # splicing sentences together). `.replace(' ','')` then turns
                # that decode into "", `word_len` never advances, and the
                # while loop keeps consuming tokens meant for *later* words
                # without bound — eventually running `k` past the end of
                # `token_ids` (IndexError, hit for real on a squad_v2
                # injected_context sentence with mdeberta-v3-base). Once out
                # of real tokens, stop rather than crash: this word (and any
                # after it) ends up with fewer/no mapped tokens, but
                # `get_token_dependency_tree`'s inner joins already drop
                # unmatched rows, and callers already tolerate/re-filter
                # sentence-vs-tree token-count mismatches (see
                # classifier_explainer.py's `_rebase_dependency_tree` and its
                # sentence-mismatch handling in `explain_context`).
                log.warning(
                    "compute_position_mapping ran out of tokens (%d) while still "
                    "matching word %d/%d (%r) in sentence=%r — likely a "
                    "whitespace-only subtoken upstream throwing off the "
                    "length budget; truncating instead of crashing.",
                    len(token_ids), i + 1, len(words), word, sentence,
                )
                break
            decoded_word = tokenizer.decode([token_ids[k]]).replace(' ','')
            # PIArena fix: the original check only recognized BPE-style special
            # tokens like "<s>"/"</s>" (GPT-2/Mistral). DeBERTa-v3 tokenizers
            # (used by meta-llama/Prompt-Guard-86M) use bracket-style special
            # tokens like "[CLS]"/"[SEP]" instead — without this branch those
            # get miscounted as regular word characters and the word<->token
            # position mapping drifts out of alignment (see Risco #1 do plano).
            is_special_token = (
                (decoded_word.startswith('<') and decoded_word.endswith('>'))
                or (decoded_word.startswith('[') and decoded_word.endswith(']'))
            )
            if is_special_token:
                # PIArena fix #2: a special token doesn't belong to *any* real
                # word — mapping it to whichever word index the outer loop
                # happened to be on (the original behavior, once the length-
                # budget bug above is fixed) still makes it spuriously merge
                # into that word's dependency-tree row downstream, because
                # get_token_dependency_tree's joins key on `word_position`
                # only, not on the token itself (confirmed empirically: [CLS]
                # was leaking in as a duplicate of the sentence's first word).
                # -1 matches no real df_words row, so the inner join in
                # get_token_dependency_tree naturally drops it instead.
                pos_token_to_word[k] = -1
            else:
                word_len += len(decoded_word)
                pos_token_to_word[k] = i
            k += 1

    return df_words, df_tokens, pos_token_to_word

def get_token_dependency_tree(sentence, tokenizer):
    """
    Get the token dependency tree DataFrame for a given sentence.

    Args:
        sentence (str): Input sentence.
        tokenizer: Tokenizer object.

    Returns:
        pd.DataFrame: Token dependency tree DataFrame.
    """
    # Single spaCy parse of `sentence`, shared by both the word list used
    # for tokenizer<->word alignment and the dependency tree itself — see
    # spacy_parse_words_and_tree's docstring for why using two independently
    # -derived word segmentations here used to silently produce empty/
    # near-empty trees on real (punctuation/contraction-heavy) sentences.
    words, tree_df = spacy_parse_words_and_tree(sentence)

    df_words, df_tokens, pos_token_to_word = compute_position_mapping(words, sentence, tokenizer)
    # Add 'word' column based on token_id_to_word mapping
    df_tokens['word_position'] = df_tokens['token_position'].map(pos_token_to_word)

    # Merge based on 'word'
    merged_df = pd.merge(df_tokens, df_words, on='word_position', how='inner')

    dependency_tree = pd.merge(merged_df, tree_df, on=['word', 'word_position'], how='inner')
    return dependency_tree

