
DECOMPOSE_CLAIM_PROMPT = ["""
You are an expert in breaking down a sentence into subclaims.
Critical Rules:
1. Each subclaim should be decontextualized from the original claim and independent from other subclaims, or self-contained, in other words. This means no pronouns should be used in place of names, and each subclaim should be unambiguous when examined in isolation.
2. You should breakdown the claim, but focus on core logical components, not pedantic details, and don't introduce pedantic subclaims that are not necessary.
3. Do not overdecompose, e.g. needlessly seperate parts of the claim which are logically connected. Not everything needs to be as atomic as possible. Less is more, and do not make subclaims that are not necessary, most subclaims are directly stated or implied by the text.
4. Subclaims should be decontextualized, but other than that, they should be as close to the language of the original claim as possible, taking as close to snippets from the original claim as possible. with the potential exception of subclaims that were strongly implied by the text.
5. Decomposition primarily serves to simply break down the claim into smaller subclaims, to make it easier to verify, overdecomposition is counterproductive.

Example:
Context: Aspirin was first synthesized over a century ago by chemist Felix Hoffmann at Bayer.
Claim: Aspirin was first synthesized over a century ago by chemist Felix Hoffmann at Bayer.
Decomposed subclaims:
1. Aspirin was first synthesized over a century ago
2. Felix Hoffman was a chemist
3. Aspirin was synthesized by Felix Hoffman at Bayer
""", """Context: {}
Claim: {}
Decomposed subclaims:
"""
]

DECONTEXTUALIZE_SENTENCE_PROMPT = ["""
You are an expert at decontextualizing sentences to make them self-contained and unambiguous.

Critical Rules:
1. Replace all pronouns (he, she, it, they, this, that, etc.) with the specific names or entities they refer to.
2. Add any necessary context that was implied by the surrounding text.
3. Make the sentence completely self-contained so it can be understood in isolation.
4. Keep the sentence as close to the original language as possible while making it clear.
5. Do not add information that is not implied by the original sentence and its context.

Example:
Context: "John Smith is a scientist. He works at MIT. He discovered a new compound."
Sentence: "He discovered a new compound."
Decontextualized: "John Smith discovered a new compound."

Context: "The company reported record profits this quarter. This represents a 20% increase."
Sentence: "This represents a 20% increase."
Decontextualized: "The company's record profits this quarter represents a 20% increase."
""", """Context:
{}
END OF CONTEXT

Sentence:
{}

Decontextualized sentence:
"""
]

CLASSIFY_SUBCLAIM_PROMPT = ["""You are an expert at classifying sentences based on their relationship to the provided context and the subject(s) of that context.

Definitions:
Subject(s) of the source: the main entities or events the context is about.

Classifications:
NON_VERIFIABLE: Contains opinions or judgments, and background/common-knowledge or definite-truth statements not about the subject(s) of the source. Includes math or logic truths, calendar arithmetic, unit conversions, definitional or taxonomic facts, and geographic containment that do not need verification against the context. These are often bridge facts used to connect evidence.
EXTRACTIVE: Contains claims that are directly supported or directly refuted by explicit spans in the context without reasoning.
INFERENTIAL: Contains claims about the subject(s) of the source that are not directly supported or refuted by the context and require multi-hop reasoning over the provided evidence. They may rely on NON_VERIFIABLE background facts as bridges, but the claim itself is about the subject(s).

Rubric:
1) Identify the subject(s) of the source.
2) If the ENTIRE claim is a background or definite-truth proposition not about the subject(s) of the source, classify as NON_VERIFIABLE.
3) Else, if explicit context spans support or refute the claim, classify as EXTRACTIVE.
4) Else, classify as INFERENTIAL.
Tie-breakers:
- Prefer NON_VERIFIABLE for math, logic, calendar arithmetic, unit conversions, definitional or lexical truths, and geography containment that are not about the subject(s).
- Do not mark as NON_VERIFIABLE if the statement asserts a property or relation of the subject(s), even if widely known; that is INFERENTIAL unless directly supported by the context.
- If deciding requires external, subject-specific facts not in the context, classify as INFERENTIAL.

1. First reason toward your decision. Do not decide until after you have reasoned.
2. After reasoning, output exactly one label from {NON_VERIFIABLE, EXTRACTIVE, INFERENTIAL} on a new line and nothing else.
""","""Context:
{}
END OF CONTEXT

Claim:
{}

Let's think step by step:
"""
]

COLLECT_NEXT_EVIDENCE_PROMPT = [
    """You are an expert at extracting evidence from context to support or refute a subclaim.\n"""
    "Critical Rules:\n"
    "1. If possible, extract the span of evidence that is most directly relevant to the subclaim.\n"
    "2. Don't repeat evidence that has already been collected.\n"
    "3. If there is truly no additional relevant evidence in the context, output the token <NO_MORE_EVIDENCE>.\n",
    """Context:\n{}\nEND OF CONTEXT\n\nSubclaim:\n{}\n\nAlready collected evidence (do not repeat):\n{}\n\nNext evidence or <NO_MORE_EVIDENCE>: """
]

PROPOSE_SUPPORTING_FACTS_PROMPT = ["""
You are an expert at constructing logical bridges between evidence and an inferential subclaim to either support or refute the subclaim.

Terminology:
- EVIDENCE fact: directly supported by explicit spans in the context.
- BACKGROUND fact: common knowledge or definite truth (math, logic, calendar arithmetic, definitions, geography containment) that is not about the subject(s) of the source and does not require verification against the context. Use only if needed to connect evidence to the subclaim.

Critical Rules:
1. Include EVIDENCE facts only if they are explicitly supported by the context. Closely paraphrase or directly copy the supporting span.
2. You may include BACKGROUND facts that are not about the subject(s) and are necessary to form the reasoning chain. Do not introduce subject-specific facts that are absent from the context.
3. Order the facts so they form a minimal, coherent chain that best supports or refutes the subclaim.
4. Do not add new, subject-specific information. If the context provides nothing usable, output the token <NO_SUPPORTING_FACTS>.
5. Reason first, then output the FACTS.
6. Don't be picky.

Example:
Context:
"Aspirin was first synthesized in 1897 by chemist Felix Hoffmann at Bayer."
END OF CONTEXT

Inferential subclaim:
"Aspirin was synthesized over a century ago"

Let's think step by step:
From the context we know the synthesis year is 1897. Using current-year arithmetic, 1897 is more than 100 years before 2025, so the subclaim is supported.

FACTS:
1. Aspirin was first synthesized in 1897
2. The current year is 2025.
3. 1897 is more than 100 years before 2025.

""","""CONTEXT:
{}
END OF CONTEXT

Inferential subclaim:
{}

Let's think step by step:
"""]

JUDGE_INF_CLAIM_SUPPORT_PROMPT = ["""
You are an expert at judging whether a set of proposed supporting facts logically supports an inferential subclaim.

Critical Rules:
1. Use only the facts provided; do not rely on any external knowledge or assumptions except for cases of common knowledge or facts that need not be verified.
2. The supporting facts should be able to form a coherent reasoning chain that directly supports the subclaim.
3. Output sections in this order: Reasoning, then final judgment ("YES" or "NO"). YES for supported, NO for refuted.
4. Don't be pedantic in your judgements, direct contradictions or completely unfounded statements are mainly what we seek to prevent. Refuted claims should be clearly, strongly refutable.

Example:
Context:
"Aspirin was first synthesized in 1897 by chemist Felix Hoffmann at Bayer."
END OF CONTEXT

Supporting facts:
1) [EVIDENCE] "Aspirin was first synthesized in 1897 ..."
2) [BACKGROUND] The current year is 2025.
3) [BACKGROUND] 1897 is more than 100 years before 2025.
END OF SUPPORTING FACTS

Inferential subclaim:
"Aspirin was synthesized over a century ago"

Let's think step by step:
The facts provide the synthesis year, the current year, and the difference being more than 100 years. This supports the the subclaim.

Is the claim supported?:
YES

""","""Context:
{}
END OF CONTEXT

Supporting facts:
{}
END OF SUPPORTING FACTS

Inferential subclaim:
{}

Let’s think step-by-step:
"""]

JUDGE_CLAIM_SUPPORT_PROMPT = [
    """You are an expert at evaluating whether a claim is fully supported by the provided evidence.
Critical Rules:
1. The claim is only supported if each part is supported by the evidence.
2. First reason towards your decision, and do not make up your mind until after you have reasoned.
3. Once done reasoning, return "YES" on a new line if the claim is supported, otherwise "NO".
4. Additionally, the evidence is known to be relevant to the claim, so you shouldn't be picky about certain ambiguities (such as using only last names).
5. Do not be overly picky, we are mainly looking to detect contradictions or things completely unsupported so don't be strict about mildly different interpretations, and don't be picky about common knowledge either.
6. If the claim is non-verifiable, eg (Opinions, judgments, common knowledge not related to the passage, or other statements not needing verification), then treat it as supported.
""",
    """
Example:
Evidence from the passage:
["On Monday, Clara Johnson, 27, and her friend, Maya Patel, 29, were detained in Los Angeles and accused of plotting to smuggle illegal weapons into the country, federal prosecutors said.",
 "The FBI charged a Houston man on Monday with attempting to provide material support to a foreign militia. He was one of three people arrested this week on weapons-trafficking charges. Two Los Angeles women were also taken into custody."]
END OF EVIDENCE

Claim:
The arrest of Jamal Carter in Houston on charges of attempting to aid a foreign militia followed the arrests of two Los Angeles women, Clara Johnson and Maya Patel, who were accused of plotting to smuggle illegal weapons into the United States.

Reasoning:
Let's first break the claim into components:
1. Jamal Carter was arrested in Houston on charges of attempting to aid a foreign militia.
2. This arrest came after the arrests of two Los Angeles women, Clara Johnson and Maya Patel.
3. Johnson and Patel were accused of plotting to smuggle illegal weapons into the United States.

Now match each component to the evidence:
- Evidence line 2 states the FBI charged a Houston man on Monday with attempting to provide material support to a foreign militia. The man is unnamed, but given the evidence is known to be relevant, we can reasonably link this Houston man to Jamal Carter. → supports component 1.
- Evidence line 2 also says the Houston man was “one of three people arrested this week,” and that “two Los Angeles women were also taken into custody,” aligning with the claim that his arrest followed theirs, even if the temporal order is not explicitly stated, that is an insignificant detail. → supports component 2.
- Evidence line 1 explicitly reports that Clara Johnson and Maya Patel were detained in Los Angeles and accused of plotting to smuggle illegal weapons. → supports component 3.

All parts of the claim are supported for; there are no contradictions or wildly unsupported components.

YES


Evidence from the passage:
{}
END OF EVIDENCE

Claim:
{}

Reasoning:"""
]

# Baseline prompt for direct hallucination detection
BASELINE_HALLUCINATION_PROMPT = ["""
Your task is to check if the Summary is accurate to the Evidence.
Generate ’Supported’ if the Summary is supported when verified according to the Evidence,
or ’Unsupported’ if the Summary is inaccurate (contradicts the evidence) or cannot be
verified.
**Evidence**\n\n{}\n\n**End of Evidence**\n
**Summary**:\n\n{}\n\n**End of Summary**\n
Classification ('Supported' or 'Unsupported'):
"""]