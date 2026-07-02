from llm_utils import *
from prompts import *
from tree_node import TreeNode, NodeType
from typing import Optional, List
from lettucedetect.models.inference import HallucinationDetector
import logging
from nltk.tokenize import sent_tokenize

class ClaimTree:
    def __init__(self, claim: str, context: str):
        self.llm_utils = create_llm_utils()
        self.claim = claim
        self.context = context
        self.root = TreeNode(NodeType.ROOT, info=claim)
        self.detector = HallucinationDetector(
            method="transformer",
            model_path="KRLabsOrg/lettucedect-large-modernbert-en-v1"
        )

    def decompose_claim(self, claim: str, sentence: str) -> List[str]:
        decomposed_str = generate(self.llm_utils, DECOMPOSE_CLAIM_PROMPT[0], DECOMPOSE_CLAIM_PROMPT[1].format(claim, sentence))
        subclaims = []
        for line in decomposed_str.splitlines():
            line = line.lstrip()
            if line and len(line) > 2 and line[0].isdigit():
                line = line[1:]
                line = line.strip('. -')
            if line:
                subclaims.append(line.strip())
        return subclaims

    def get_sentences(self, text: str) -> List[str]:
        return sent_tokenize(text)

    def decontextualize_sentence(self, sentence: str) -> str:
        """Decontextualize a sentence to make it self-contained using the full claim as context."""
        decontextualized = generate(self.llm_utils, DECONTEXTUALIZE_SENTENCE_PROMPT[0], DECONTEXTUALIZE_SENTENCE_PROMPT[1].format(self.claim, sentence))
        
        # Extract the decontextualized sentence from the response
        lines = decontextualized.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line and not line.startswith(('Context:', 'Sentence:', 'Decontextualized sentence:')):
                return line
        
        # Fallback: return original sentence if parsing fails
        return sentence

    def classify_subclaim(self, subclaim: str) -> Optional[NodeType]:
        classification = generate(self.llm_utils, CLASSIFY_SUBCLAIM_PROMPT[0], CLASSIFY_SUBCLAIM_PROMPT[1].format(self.context, subclaim))
        
        # Extract the classification from the last line due to chain of thought reasoning
        lines = classification.strip().split('\n')
        last_line = lines[-1].strip() if lines else ""
        
        # Check the last line for the classification
        if "NON_VERIFIABLE" in last_line:
            return None
        elif "EXTRACTIVE" in last_line:
            return NodeType.EXTRACTIVE
        elif "INFERENTIAL" in last_line:
            return NodeType.INFERENTIAL
        
        # Fallback: check the entire response if last line doesn't contain classification
        if "NON_VERIFIABLE" in classification:
            return None
        elif "EXTRACTIVE" in classification:
            return NodeType.EXTRACTIVE
        elif "INFERENTIAL" in classification:
            return NodeType.INFERENTIAL
        
        # If no classification found, return None
        return None

    def construct(self) -> bool:
        #self.subclaims = self.decompose_claim(self.claim)                                                      #### USING SENTENCES
        self.sentences = self.get_sentences(self.claim)
        self.subclaims = []
        for sentence in self.sentences:
            decontextualized_sentence = self.decontextualize_sentence(sentence)
            self.subclaims.append(decontextualized_sentence)
        for subclaim in self.subclaims:
            if not self.verify_subclaim(self.root, subclaim):
                self.root.supported = False
                return False
        return True

    def verify_subclaim(self, parent_node: TreeNode, subclaim: str) -> bool:
        classification = self.classify_subclaim(subclaim)
        if classification is None:
            return True
        node = parent_node.add_child(TreeNode(classification, info=subclaim))
        if classification == NodeType.EXTRACTIVE:
            node.supported = self.verify_claim_with_lettuce(subclaim)
        elif classification == NodeType.INFERENTIAL:
            node.supported = self.verify_inferential_subclaim(subclaim, node)
        return node.supported

    def collect_next_evidence(self, subclaim: str, current_evidence: List[str]) -> str:
        return generate(self.llm_utils, COLLECT_NEXT_EVIDENCE_PROMPT[0], COLLECT_NEXT_EVIDENCE_PROMPT[1].format(self.context, subclaim, str(current_evidence)))

    def collect_supporting_facts(self, subclaim: str) -> List[str]:
        supporting_facts = generate(self.llm_utils, PROPOSE_SUPPORTING_FACTS_PROMPT[0], PROPOSE_SUPPORTING_FACTS_PROMPT[1].format(self.context, subclaim))
        if "<NO_SUPPORTING_FACTS>" in supporting_facts:
            return []
        
        facts = []
        lines = supporting_facts.splitlines()
        in_facts_section = False
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if we've reached the FACTS section
            if line.upper() == "FACTS:":
                in_facts_section = True
                continue
            
            # If we're in the FACTS section, parse numbered items
            if in_facts_section:
                # Check for numbered list items (1., 2., etc.)
                if line and len(line) > 2 and line[0].isdigit():
                    fact = line.strip(' .-')
                    if fact:
                        facts.append(fact)
                # Also check for lines that might be facts without numbering
                elif line and not line.startswith(('Let\'s think step by step:', 'FACTS:')):
                    facts.append(line)
        
        # Fallback: if no FACTS section found, try the old parsing method
        if not facts:
            for line in supporting_facts.splitlines():
                line = line.lstrip()
                if line and len(line) > 2 and line[0].isdigit() and line[1:3] == '. ':
                    line = line[3:]
                if line and not line.startswith(('Let\'s think step by step:', 'FACTS:')):
                    facts.append(line.strip())
        
        return facts

    def judge_inf_claim_support(self, subclaim: str, proposed_facts: List[str]) -> bool:
        support = generate(self.llm_utils, JUDGE_INF_CLAIM_SUPPORT_PROMPT[0], JUDGE_INF_CLAIM_SUPPORT_PROMPT[1].format(self.context, str(proposed_facts), subclaim))
        
        # Extract the answer from the last line due to chain of thought reasoning
        lines = support.strip().split('\n')
        last_line = lines[-1].strip() if lines else ""
        
        # Check the last line for the answer
        if "YES" in last_line.upper():
            return True
        elif "NO" in last_line.upper():
            return False
        
        # Fallback: check the entire response if last line doesn't contain answer
        if "YES" in support.upper():
            return True
        elif "NO" in support.upper():
            return False
        
        # If no clear answer found, default to False (not supported)
        return False
    
    '''def judge_claim_support(self, subclaim: str, current_evidence: List[str]) -> bool:
        support = generate(self.llm_utils, JUDGE_CLAIM_SUPPORT_PROMPT[0], JUDGE_CLAIM_SUPPORT_PROMPT[1].format(str(current_evidence), subclaim))
        # Look for the final YES/NO at the end of the response after chain-of-thought reasoning
        lines = support.strip().split('\n')
        for line in reversed(lines):
            line = line.strip()
            if "YES" in line:
                return True
            elif "NO" in line:
                return False
        # Fallback to original behavior if no clear YES/NO found
        return "YES" in support'''
    
    def verify_extractive_with_curated_evidence_lettuce(self, node, subclaim):
        evidence = []          
        max_evidence = 6
        while len(evidence) < max_evidence:                      # CHANGE TO AND NOT SUFFICIENT  LATER
            next_evidence = self.collect_next_evidence(subclaim, evidence)
            if "<NO_MORE_EVIDENCE>" in next_evidence:
                break
            evidence.append(next_evidence)
            node.add_child(TreeNode(NodeType.EVIDENCE, info=next_evidence))
        predictions = self.detector.predict(
            context=evidence,
            question="",
            answer=subclaim,
            output_format="spans"
        )

        logging.info(f"Lettuce Verification - Claim: {subclaim}: {str(predictions)}")        
    
        return len(predictions) == 0

    def verify_inferential_subclaim(self, subclaim: str, node: TreeNode) -> bool:
        supporting_facts = self.collect_supporting_facts(subclaim)
        if supporting_facts == []:
            return False
        for fact in supporting_facts:
            node.add_child(TreeNode(NodeType.EVIDENCE, fact))
        return self.judge_inf_claim_support(subclaim, supporting_facts)

    def print_tree(self, node: Optional[TreeNode] = None, indent: int = 0):
        if node is None:
            node = self.root
        prefix = "    " * indent
        if node.supported is not None:
            if node.supported:
                supported_str = " [SUPPORTED]"
            else:
                supported_str = " [NOT_SUPPORTED]"
        else:
            supported_str = ""
        print(f"{prefix}{node.node_type.value}{supported_str}: {node.info}")
        for child in node.children:
            self.print_tree(child, indent + 1)

    def baseline(self) -> bool:
        response = generate_no_system(self.llm_utils,
                BASELINE_HALLUCINATION_PROMPT[0].format(self.context, self.claim)
            )
            
        # Parse response
        return "yes" in response.lower()

    def verify_claim_with_lettuce(self, claim: str) -> bool:
        predictions = self.detector.predict(
            context=[self.context],
            question="",
            answer=claim,
            output_format="spans"
        )

        logging.info(f"Lettuce Verification - Claim: {claim}: {str(predictions)}")        
    
        return len(predictions) == 0 # If predictions is empty, the claim is supported (no hallucination)
    
    def verify_inf_claim_with_lettuce(self, claim: str, supporting_facts: List[str]) -> bool:
        predictions = self.detector.predict(
            context=[self.context] + supporting_facts,
            question="",
            answer=claim,
            output_format="spans"
        )

        logging.info(f"Lettuce Verification - Claim: {claim}: {str(predictions)}")        
    
        return len(predictions) == 0 # If predictions is empty, the claim is supported (no hallucination)

    # Ensure all is robust for chain of thought outputs