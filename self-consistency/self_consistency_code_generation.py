
import os
import json
import openai
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
import re
import warnings
warnings.filterwarnings('ignore')
from dataclasses import dataclass
# Clustering imports
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
import os
import json
import openai
from typing import List, Dict, Any, Optional
from collections import Counter
import re
import os
import json
from sentence_transformers import SentenceTransformer
import openai
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
import re
import warnings
warnings.filterwarnings('ignore')

# Clustering imports
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def __add__(self, other):
        return TokenUsage(self.prompt_tokens + other.prompt_tokens, self.completion_tokens + other.completion_tokens, self.total_tokens + other.total_tokens)


class CodeEmbedder:
    """
    Code embedding wrapper that can use various embedding models
    """

    def __init__(self, model_name: str = "microsoft/codebert-base", device: str = 'cpu'):
        """
        Initialize code embedder with specified model

        Recommended code embedding models:
        - "all-MiniLM-L6-v2": General purpose, fast
        - "krlvi/sentence-t5-base-nlpl-code_search_net": Code-specific
        - "microsoft/codebert-base": CodeBERT for code understanding
        - "jinaai/jina-embeddings-v2-base-code": Jina code embeddings

        Args:
            model_name: Name/path of the embedding model
            device: Device to run model on ('cuda', 'cpu', etc.)
        """
        self.model_name = model_name
        self.device = device

        try:
            self.model = SentenceTransformer(model_name, device='cpu')
            # self.model.to(device)
            print(f"🤖 Code Embedder initialized: {model_name}")

        except Exception as e:
            raise e
            print(f"❌ Failed to load embedding model {model_name}: {e}")
            print("    Using fallback mock embedder")
            self.model = None
            self.embedding_dim = 384

    def encode(self, code_snippets: List[str]) -> np.ndarray:
        """
        Encode code snippets into embeddings

        Args:
            code_snippets: List of code strings

        Returns:
            numpy array of embeddings (n_samples, embedding_dim)
        """
        if self.model is not None:
            # Real embedding model
            return self.model.encode(code_snippets)
        else:
            # Mock embedding for demonstration
            return self._mock_encode(code_snippets)

    def _mock_encode(self, code_snippets: List[str]) -> np.ndarray:
        """Generate mock embeddings based on code features"""
        embeddings = []

        for code in code_snippets:
            # Extract meaningful code features
            features = []

            # Basic structural features
            features.extend([
                len(code),                              # code length
                code.count('\n'),                       # number of lines
                code.count(' '),                        # whitespace (complexity)
                code.count('def '),                     # function definitions
                code.count('class '),                   # class definitions
                code.count('for '),                     # for loops
                code.count('while '),                   # while loops
                code.count('if '),                      # conditionals
                code.count('elif '),                    # elif statements
                code.count('else'),                     # else statements
                code.count('return '),                  # return statements
                code.count('import '),                  # imports
                code.count('from '),                    # from imports
                code.count('try:'),                     # try blocks
                code.count('except'),                   # exception handling
                code.count('='),                        # assignments
                code.count('=='),                       # comparisons
                code.count('('),                        # function calls/grouping
                code.count('['),                        # list/array operations
                code.count('{'),                        # dict operations
                code.count('+='),                       # augmented assignment
                code.count('and '),                     # logical and
                code.count('or '),                      # logical or
                code.count('not '),                     # logical not
            ])

            # Algorithm-specific patterns
            algorithms = [
                'expand', 'center', 'around',          # expand around center
                'dp', 'dynamic', 'programming',        # dynamic programming
                'brute', 'force', 'nested',            # brute force
                'recursive', 'recursion',              # recursive approaches
                'iterative', 'iteration',              # iterative approaches
                'two', 'pointer', 'pointers',          # two pointer technique
                'sliding', 'window',                   # sliding window
                'hash', 'map', 'dict',                 # hash-based solutions
                'sort', 'sorted',                      # sorting-based
                'binary', 'search',                    # binary search
                'greedy',                              # greedy algorithms
                'backtrack',                           # backtracking
                'divide', 'conquer',                   # divide and conquer
            ]

            code_lower = code.lower()
            for alg in algorithms:
                features.append(1 if alg in code_lower else 0)

            # Variable naming patterns
            common_vars = ['i', 'j', 'k', 'n', 'len', 'start', 'end', 'left', 'right',
                          'result', 'temp', 'curr', 'prev', 'next', 'max', 'min']
            for var in common_vars:
                features.append(1 if f' {var} ' in code or f'{var}=' in code else 0)

            # Pad or truncate to target dimension
            while len(features) < self.embedding_dim:
                features.append(0.0)
            features = features[:self.embedding_dim]

            # Add some noise for variety but keep it small
            features = np.array(features, dtype=float)
            noise = np.random.normal(0, 0.05, self.embedding_dim)
            features = features + noise

            embeddings.append(features)

        return np.array(embeddings)


def compute_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """
    Compute pairwise similarity matrix using cosine similarity

    Args:
        embeddings: Code embeddings matrix (n_samples, embedding_dim)

    Returns:
        Similarity matrix (n_samples, n_samples)
    """
    return cosine_similarity(embeddings)


def find_optimal_clusters_hierarchical(
    similarity_matrix: np.ndarray,
    max_clusters: int = None,
    linkage_methods: List[str] = None
) -> Tuple[np.ndarray, Dict]:
    """
    Find optimal hierarchical clustering configuration

    Args:
        similarity_matrix: Pairwise similarity matrix
        max_clusters: Maximum number of clusters to consider
        linkage_methods: List of linkage methods to try

    Returns:
        Best cluster labels and metadata
    """
    n_samples = similarity_matrix.shape[0]
    if max_clusters is None:
        max_clusters = min(n_samples // 2, 8)

    if linkage_methods is None:
        linkage_methods = ['complete', 'average', 'single']

    # Convert similarity to distance
    distance_matrix = 1 - similarity_matrix
    distance_matrix = np.clip(distance_matrix, 0, 2)

    best_score = -1
    best_labels = None
    best_params = {}

    for n_clusters in range(2, min(max_clusters + 1, n_samples)):
        for linkage in linkage_methods:
            try:
                clustering = AgglomerativeClustering(
                    n_clusters=n_clusters,
                    linkage=linkage,
                    metric='precomputed'
                )
                labels = clustering.fit_predict(distance_matrix)

                # Score based on cluster balance and compactness
                unique_labels, counts = np.unique(labels, return_counts=True)

                # Prefer balanced clusters
                balance_score = 1.0 / (1.0 + np.std(counts) / np.mean(counts))

                # Prefer moderate number of clusters
                cluster_penalty = abs(n_clusters - n_samples / 4) / n_samples
                cluster_score = 1.0 / (1.0 + cluster_penalty)

                # Compute average intra-cluster similarity
                intra_sim = 0
                for label in unique_labels:
                    cluster_mask = (labels == label)
                    cluster_indices = np.where(cluster_mask)[0]
                    if len(cluster_indices) > 1:
                        cluster_sim = similarity_matrix[np.ix_(cluster_indices, cluster_indices)]
                        # Average of upper triangle (exclude diagonal)
                        upper_tri = cluster_sim[np.triu_indices_from(cluster_sim, k=1)]
                        intra_sim += np.mean(upper_tri) if len(upper_tri) > 0 else 0

                intra_sim /= len(unique_labels)

                score = balance_score * cluster_score * (1 + intra_sim)

                if score > best_score:
                    best_score = score
                    best_labels = labels
                    best_params = {
                        'n_clusters': n_clusters,
                        'linkage': linkage,
                        'score': score,
                        'balance_score': balance_score,
                        'cluster_score': cluster_score,
                        'intra_similarity': intra_sim
                    }

            except Exception:
                continue

    # Fallback handling
    if best_labels is None:
        try:
            clustering = AgglomerativeClustering(n_clusters=2, linkage='complete', metric='precomputed')
            best_labels = clustering.fit_predict(distance_matrix)
            best_params = {'n_clusters': 2, 'linkage': 'complete', 'score': 0, 'fallback': True}
        except Exception:
            best_labels = np.zeros(n_samples, dtype=int)
            best_params = {'n_clusters': 1, 'score': 0, 'ultimate_fallback': True}

    return best_labels, best_params


def find_optimal_clusters_dbscan(
    similarity_matrix: np.ndarray,
    eps_range: Tuple[float, float] = (0.1, 0.8),
    min_samples_range: Tuple[int, int] = (2, 5)
) -> Tuple[np.ndarray, Dict]:
    """
    Find optimal DBSCAN clustering configuration

    Args:
        similarity_matrix: Pairwise similarity matrix
        eps_range: Range of eps values to try
        min_samples_range: Range of min_samples values to try

    Returns:
        Best cluster labels and metadata
    """
    # Convert similarity to distance
    distance_matrix = 1 - similarity_matrix
    distance_matrix = np.clip(distance_matrix, 0, 2)

    best_score = -1
    best_labels = None
    best_params = {}

    # Grid search
    eps_values = np.linspace(eps_range[0], eps_range[1], 10)
    min_samples_values = range(min_samples_range[0], min_samples_range[1] + 1)

    for eps in eps_values:
        for min_samples in min_samples_values:
            try:
                dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
                labels = dbscan.fit_predict(distance_matrix)

                unique_labels = set(labels)
                n_clusters = len([label for label in unique_labels if label != -1])
                n_noise = sum(1 for label in labels if label == -1)

                # Skip poor clusterings
                if n_clusters < 1:
                    continue

                # Score based on cluster quality
                noise_ratio = n_noise / len(labels)
                cluster_score = n_clusters if n_clusters <= len(labels) // 2 else 0

                # Prefer low noise and reasonable cluster count
                score = cluster_score * (1 - noise_ratio * 0.7)

                if score > best_score:
                    best_score = score
                    best_labels = labels
                    best_params = {
                        'eps': eps,
                        'min_samples': min_samples,
                        'score': score,
                        'n_clusters': n_clusters,
                        'noise_ratio': noise_ratio
                    }

            except Exception:
                continue

    # Fallback
    if best_labels is None:
        try:
            dbscan = DBSCAN(eps=0.5, min_samples=2, metric='precomputed')
            best_labels = dbscan.fit_predict(distance_matrix)
            best_params = {'eps': 0.5, 'min_samples': 2, 'score': 0, 'fallback': True}
        except Exception:
            best_labels = np.zeros(distance_matrix.shape[0], dtype=int)
            best_params = {'score': 0, 'ultimate_fallback': True}

    return best_labels, best_params


class SelfConsistencyCodeGeneration:
    """
    Implementation of Self-Consistency approach for code generation
    Based on the paper "Self-Consistency Improves Chain of Thought Reasoning in Language Models"

    The key idea is to:
    1. Sample multiple diverse reasoning paths instead of greedy decoding
    2. Generate code solutions from each path
    3. Select the most consistent solution through majority voting
    """

    def __init__(self, api_key: str, embedder, model: str = "o3-mini", temperature: float = 0.7, top_k: int = 40):
        """
        Initialize the self-consistency code generator

        Args:
            api_key: OpenAI API key
            model: Model to use (o3-mini by default)
            temperature: Sampling temperature for diversity
            top_k: Top-k sampling parameter
        """
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.top_k = top_k
        self.embedder = embedder
        self.token_usage = TokenUsage()

    def aggregate_solutions_majority_vote_improved(
        self,
        solutions: List[Dict[str, Any]],
        clustering_method: str = "hierarchical",
        similarity_threshold: float = 0.7
    ) -> Dict[str, Any]:
        """
        Improved majority vote using embeddings and advanced clustering

        Args:
            solutions: List of solution dictionaries
            clustering_method: 'hierarchical', 'dbscan', or 'both'
            similarity_threshold: Fallback similarity threshold

        Returns:
            Aggregated solution with metadata
        """
        if not solutions:
            return {"code": "", "reasoning": "", "confidence": 0.0,
                   "metadata": {"error": "No solutions provided"}}

        # Extract valid codes
        codes = []
        valid_indices = []

        for i, sol in enumerate(solutions):
            code = self.extract_final_answer(sol)
            if code and code.strip():
                codes.append(code)
                valid_indices.append(i)

        if not codes:
            return {"code": "", "reasoning": "", "confidence": 0.0,
                   "metadata": {"error": "No valid code found"}}

        if len(codes) == 1:
            return {
                "code": codes[0],
                "reasoning": solutions[valid_indices[0]].get('reasoning', ''),
                "confidence": 1.0,
                "metadata": {"single_solution": True}
            }

        print(f"🔍 Analyzing {len(codes)} solutions with embedding-based clustering...")

        # Generate embeddings
        try:
            embeddings = self.embedder.encode(codes)
            print(f"✅ Generated embeddings: {embeddings.shape}")
        except Exception as e:
            print(f"❌ Embedding failed: {e}")
            return self._fallback_original_method(solutions)

        # Compute similarity matrix
        similarity_matrix = compute_similarity_matrix(embeddings)
        print(f"✅ Similarity matrix: {similarity_matrix.shape}, "
              f"range: [{similarity_matrix.min():.3f}, {similarity_matrix.max():.3f}]")

        # Apply clustering
        clustering_results = {}

        if clustering_method in ['hierarchical', 'both']:
            try:
                hier_labels, hier_params = find_optimal_clusters_hierarchical(similarity_matrix)
                clustering_results['hierarchical'] = {
                    'labels': hier_labels,
                    'params': hier_params,
                    'n_clusters': len(np.unique(hier_labels))
                }
                print(f"✅ Hierarchical: {clustering_results['hierarchical']['n_clusters']} clusters")
            except Exception as e:
                print(f"⚠️  Hierarchical clustering failed: {e}")

        if clustering_method in ['dbscan', 'both']:
            try:
                dbscan_labels, dbscan_params = find_optimal_clusters_dbscan(similarity_matrix)
                clustering_results['dbscan'] = {
                    'labels': dbscan_labels,
                    'params': dbscan_params,
                    'n_clusters': len([l for l in np.unique(dbscan_labels) if l != -1])
                }
                print(f"✅ DBSCAN: {clustering_results['dbscan']['n_clusters']} clusters")
            except Exception as e:
                print(f"⚠️  DBSCAN clustering failed: {e}")

        # Select best clustering
        if not clustering_results:
            print('fallback clustering...')
            return self._fallback_similarity_clustering(codes, solutions, valid_indices,
                                                      similarity_matrix, similarity_threshold)

        best_method, best_labels = self._select_best_clustering(clustering_results)
        print(f"🏆 Selected {best_method} clustering")

        # Analyze clusters and select best solution
        result = self._analyze_clusters_and_select(
            best_labels, codes, solutions, valid_indices, similarity_matrix,
            best_method, clustering_results
        )

        return result

    def _select_best_clustering(self, clustering_results: Dict) -> Tuple[str, np.ndarray]:
        """Select the best clustering method based on scores"""
        best_method = None
        best_labels = None
        best_score = -1

        for method, result in clustering_results.items():
            score = result['params'].get('score', 0)
            n_clusters = result['n_clusters']

            # Prefer methods with reasonable cluster counts and good scores
            if n_clusters >= 2 and score > best_score:
                best_score = score
                best_method = method
                best_labels = result['labels']

        # Fallback to first available
        if best_labels is None:
            best_method = list(clustering_results.keys())[0]
            best_labels = clustering_results[best_method]['labels']

        return best_method, best_labels

    def _analyze_clusters_and_select(
        self,
        labels: np.ndarray,
        codes: List[str],
        solutions: List[Dict],
        valid_indices: List[int],
        similarity_matrix: np.ndarray,
        method: str,
        clustering_results: Dict
    ) -> Dict[str, Any]:
        """Analyze clusters and select the best representative solution"""

        unique_labels = np.unique(labels)
        cluster_info = []

        # Analyze each cluster
        for label in unique_labels:
            if label == -1:  # Skip noise in DBSCAN
                continue

            cluster_indices = np.where(labels == label)[0]

            # Calculate intra-cluster similarity
            if len(cluster_indices) > 1:
                cluster_similarities = []
                for i in range(len(cluster_indices)):
                    for j in range(i + 1, len(cluster_indices)):
                        sim = similarity_matrix[cluster_indices[i], cluster_indices[j]]
                        cluster_similarities.append(sim)
                avg_similarity = np.mean(cluster_similarities)
            else:
                avg_similarity = 1.0

            cluster_info.append({
                'label': int(label),
                'size': len(cluster_indices),
                'indices': cluster_indices,
                'avg_similarity': float(avg_similarity),
                'score': len(cluster_indices) * avg_similarity  # Size × quality
            })

        if not cluster_info:
            # Fallback: treat all as one cluster
            cluster_info = [{
                'label': 0,
                'size': len(codes),
                'indices': np.arange(len(codes)),
                'avg_similarity': 1.0,
                'score': len(codes)
            }]

        # Select best cluster
        best_cluster = max(cluster_info, key=lambda x: x['score'])

        # Select representative from best cluster
        representative_idx = self._select_cluster_representative(
            best_cluster['indices'], similarity_matrix
        )

        # Get original solution
        original_idx = valid_indices[representative_idx]
        selected_solution = solutions[original_idx]

        # Calculate confidence
        cluster_size_ratio = best_cluster['size'] / len(codes)
        similarity_score = best_cluster['avg_similarity']
        confidence = cluster_size_ratio * 0.6 + similarity_score * 0.4

        # Build metadata
        metadata = {
            'embedding_model': self.embedder.model_name,
            'clustering_method': method,
            'clustering_params': clustering_results[method]['params'],
            'n_total_solutions': len(codes),
            'n_clusters_found': len(cluster_info),
            'selected_cluster_size': best_cluster['size'],
            'selected_cluster_avg_similarity': best_cluster['avg_similarity'],
            'cluster_size_ratio': cluster_size_ratio,
            'all_clusters': [
                {
                    'label': c['label'],
                    'size': c['size'],
                    'avg_similarity': c['avg_similarity'],
                    'score': c['score']
                }
                for c in cluster_info
            ]
        }

        result = {
            "code": selected_solution.get('code', ''),
            "reasoning": selected_solution.get('reasoning', ''),
            "confidence": float(confidence),
            "metadata": metadata
        }

        print(f"🎯 Selected solution from cluster of size {best_cluster['size']} "
              f"with confidence {confidence:.3f}")

        return result

    def _select_cluster_representative(self, cluster_indices: np.ndarray,
                                     similarity_matrix: np.ndarray) -> int:
        """Select the most representative solution from a cluster"""
        if len(cluster_indices) == 1:
            return cluster_indices[0]

        # Select solution with highest average similarity to others in cluster
        avg_similarities = []
        for idx in cluster_indices:
            similarities_to_cluster = [
                similarity_matrix[idx, other_idx]
                for other_idx in cluster_indices if other_idx != idx
            ]
            avg_sim = np.mean(similarities_to_cluster) if similarities_to_cluster else 0
            avg_similarities.append(avg_sim)

        best_in_cluster = np.argmax(avg_similarities)
        return cluster_indices[best_in_cluster]

    def _fallback_similarity_clustering(self, codes, solutions, valid_indices,
                                      similarity_matrix, threshold):
        """Fallback method using simple similarity threshold"""
        print(f"🔄 Using similarity threshold clustering ({threshold})")

        clusters = []
        used = set()

        for i in range(len(codes)):
            if i in used:
                continue

            cluster = [i]
            used.add(i)

            for j in range(len(codes)):
                if j != i and j not in used:
                    if similarity_matrix[i, j] > threshold:
                        cluster.append(j)
                        used.add(j)

            clusters.append(cluster)

        # Select largest cluster
        best_cluster = max(clusters, key=len)
        rep_idx = best_cluster[0] if len(best_cluster) == 1 else                   best_cluster[np.argmax([
                      np.mean([similarity_matrix[idx, other] for other in best_cluster if other != idx])
                      for idx in best_cluster
                  ])]

        original_idx = valid_indices[rep_idx]
        selected_solution = solutions[original_idx]

        return {
            "code": selected_solution.get('code', ''),
            "reasoning": selected_solution.get('reasoning', ''),
            "confidence": len(best_cluster) / len(codes),
            "metadata": {
                "method": "similarity_threshold",
                "threshold": threshold,
                "n_clusters": len(clusters),
                "selected_cluster_size": len(best_cluster)
            }
        }

    def _fallback_original_method(self, solutions):
        """Ultimate fallback to original token-based method"""
        print("🔄 Using original token-based method")

        codes = [self.extract_final_answer(sol) for sol in solutions if sol.get('code')]
        if not codes:
            return {"code": "", "reasoning": "", "confidence": 0.0}

        # Simple grouping by token similarity
        code_groups = []
        similarity_threshold = 0.8

        for code in codes:
            placed = False
            for group in code_groups:
                if self.compute_code_similarity(code, group['representative']) > similarity_threshold:
                    group['codes'].append(code)
                    group['count'] += 1
                    placed = True
                    break

            if not placed:
                code_groups.append({'representative': code, 'codes': [code], 'count': 1})

        best_group = max(code_groups, key=lambda g: g['count'])

        # Find corresponding reasoning
        best_code = best_group['representative']
        best_reasoning = ""
        for sol in solutions:
            if self.compute_code_similarity(sol.get('code', ''), best_code) > similarity_threshold:
                best_reasoning = sol.get('reasoning', '')
                break

        return {
            "code": best_code,
            "reasoning": best_reasoning,
            "confidence": best_group['count'] / len(solutions),
            "metadata": {"method": "original_fallback", "n_groups": len(code_groups)}
        }

    # Main public method
    def aggregate_solutions_majority_vote(
        self,
        solutions: List[Dict[str, Any]],
        use_embeddings: bool = True,
        clustering_method: str = "hierarchical",
        similarity_threshold: float = 0.7
    ) -> Dict[str, Any]:
        """
        Main aggregation method with embedding option

        Args:
            solutions: List of solution dictionaries
            use_embeddings: Whether to use embedding-based clustering
            clustering_method: 'hierarchical', 'dbscan', or 'both'
            similarity_threshold: Fallback threshold

        Returns:
            Aggregated solution with metadata
        """
        if use_embeddings:
            return self.aggregate_solutions_majority_vote_improved(
                solutions, clustering_method, similarity_threshold
            )
        else:
            return self._fallback_original_method(solutions)
    def create_chain_of_thought_prompt(self, problem: str, examples: List[str] = None) -> str:
        """
        Create a chain-of-thought prompt for code generation

        Args:
            problem: The coding problem to solve
            examples: Optional few-shot examples

        Returns:
            Formatted prompt string
        """
        base_prompt = """You are an expert programmer. Solve the following coding problem step by step.

Think through the problem carefully:
1. Understand what the problem is asking
2. Plan your approach 
3. Consider edge cases
4. Write clean code
5. Explain your solution
6. Put the final code inside ```python and ``` tags
"""

        if examples:
            for example in examples:
                base_prompt += f"Example:\n{example}\n\n"

        base_prompt += f"Problem: {problem}\n\nSolution:"

        return base_prompt

    def generate_single_solution(self, prompt: str) -> Dict[str, Any]:
        """
        Generate a single code solution using the specified model

        Args:
            prompt: The input prompt

        Returns:
            Dictionary containing the response and metadata
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert programmer. Provide step-by-step reasoning followed by clean code. Put the final code inside ```python and ``` tags"},
                    {"role": "user", "content": prompt}
                ],
                # temperature=self.temperature,
                # max_tokens=1500,
                # top_p=0.9  # Using nucleus sampling as mentioned in the paper
            )

            content = response.choices[0].message.content
            usage = response.usage
            self.token_usage.completion_tokens += usage.completion_tokens
            self.token_usage.prompt_tokens += usage.prompt_tokens
            self.token_usage.total_tokens += usage.total_tokens
            return {
                "content": content,
                "reasoning": self.extract_reasoning(content),
                "code": self.extract_code(content),
                "success": True
            }

        except Exception as e:
            return {
                "content": "",
                "reasoning": "",
                "code": "",
                "success": False,
                "error": str(e)
            }

    def extract_reasoning(self, content: str) -> str:
        """Extract the reasoning part from the model's response"""
        # Simple heuristic: everything before the code block is reasoning
        code_pattern = r'```[\w]*\n(.*?)\n```'
        code_match = re.search(code_pattern, content, re.DOTALL)

        if code_match:
            code_start = content.find('```')
            return content[:code_start].strip()
        else:
            return content.strip()

    def extract_code(self, content: str) -> str:
        try:
            code = content.split('```')[1].replace('python', '')
        except IndexError:
            return content
        return code

    def generate_multiple_solutions(self, prompt: str, num_samples: int = 5) -> List[Dict[str, Any]]:
        """
        Generate multiple diverse solutions using sampling

        Args:
            prompt: The input prompt
            num_samples: Number of solutions to generate

        Returns:
            List of solution dictionaries
        """
        solutions = []

        print(f"Generating {num_samples} diverse solutions...")
        for i in range(num_samples):
            print(f"  Generating solution {i+1}/{num_samples}...")
            solution = self.generate_single_solution(prompt)
            # print(solution['code'])
            if solution['success']:
                solutions.append(solution)
            else:
                print(f"    Failed to generate solution {i+1}: {solution.get('error', 'Unknown error')}")

        print(f"Successfully generated {len(solutions)} solutions")
        return solutions

    def compute_code_similarity(self, code1: str, code2: str) -> float:
        """
        Compute similarity between two code snippets
        This is a simplified version - in practice, you might want more sophisticated comparison
        """
        if not code1 or not code2:
            return 0.0

        # Normalize whitespace and compare
        normalized1 = ' '.join(code1.split())
        normalized2 = ' '.join(code2.split())

        if normalized1 == normalized2:
            return 1.0

        # Simple token-based similarity
        tokens1 = set(normalized1.split())
        tokens2 = set(normalized2.split())

        if not tokens1 or not tokens2:
            return 0.0

        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)
        print(len(intersection) / len(union))
        return len(intersection) / len(union)

    def extract_final_answer(self, solution: Dict[str, Any]) -> str:
        return solution.get('code', '').strip()

    # def aggregate_solutions_majority_vote(self, solutions: List[Dict[str, Any]]) -> Dict[str, Any]:
    #     """
    #     Aggregate solutions using majority vote
    #     This implements the basic majority voting approach from the paper
    #     """
    #     if not solutions:
    #         return {"code": "", "reasoning": "", "confidence": 0.0}
    #
    #     codes = [self.extract_final_answer(sol) for sol in solutions if sol.get('code')]
    #
    #     if not codes:
    #         return {"code": "", "reasoning": "", "confidence": 0.0}
    #
    #     # Group similar codes together
    #     code_groups = []
    #     similarity_threshold = 0.5
    #
    #     for code in codes:
    #         placed = False
    #         for group in code_groups:
    #             if self.compute_code_similarity(code, group['representative']) > similarity_threshold:
    #                 group['codes'].append(code)
    #                 group['count'] += 1
    #                 placed = True
    #                 break
    #
    #         if not placed:
    #             code_groups.append({
    #                 'representative': code,
    #                 'codes': [code],
    #                 'count': 1
    #             })
    #
    #     # Find the group with the highest count (majority)
    #     best_group = max(code_groups, key=lambda g: g['count'])
    #
    #     # Find the corresponding reasoning from the original solutions
    #     best_code = best_group['representative']
    #     best_reasoning = ""
    #
    #     for sol in solutions:
    #         if self.compute_code_similarity(sol.get('code', ''), best_code) > similarity_threshold:
    #             best_reasoning = sol.get('reasoning', '')
    #             break
    #
    #     confidence = best_group['count'] / len(solutions)
    #
    #     return {
    #         "code": best_code,
    #         "reasoning": best_reasoning,
    #         "confidence": confidence,
    #         "vote_distribution": {f"group_{i}": group['count'] for i, group in enumerate(code_groups)}
    #     }

    def score_code_quality(self, code: str) -> float:
        """
        Simple code quality scoring heuristic
        In practice, this could be more sophisticated
        """
        if not code:
            return 0.0

        score = 1.0

        # Bonus for proper function definition
        if 'def ' in code:
            score += 0.2

        # Bonus for error handling
        if any(keyword in code for keyword in ['try:', 'except:', 'raise', 'assert']):
            score += 0.1

        # Bonus for comments/docstrings
        if '#' in code or 'docstring' in code.lower():
            score += 0.1

        # Penalty for very short or very long code
        lines = len(code.split('\n'))
        if lines < 3:
            score -= 0.1
        elif lines > 50:
            score -= 0.1

        # Bonus for common good practices
        if 'return' in code:
            score += 0.1

        return max(0.1, score)  # Minimum score of 0.1

    def self_consistency_generate(self, problem: str, num_samples: int = 5,
                                examples: List[str] = None) -> Dict[str, Any]:
        """
        Main method implementing self-consistency for code generation

        This follows the algorithm from the paper:
        1. Generate multiple reasoning paths using sampling
        2. Extract final answers (code) from each path
        3. Aggregate using majority vote to find most consistent answer

        Args:
            problem: The coding problem to solve
            num_samples: Number of diverse solutions to generate
            examples: Optional few-shot examples

        Returns:
            Dictionary with the final solution and metadata
        """
        # print("🚀 Starting Self-Consistency Code Generation")
        # print(f"Problem: {problem[:100]}{'...' if len(problem) > 100 else ''}")

        # Step 1: Create chain-of-thought prompt
        prompt = self.create_chain_of_thought_prompt(problem, examples)

        # Step 2: Generate multiple diverse solutions (sample diverse reasoning paths)
        solutions = self.generate_multiple_solutions(prompt, num_samples)

        if not solutions:
            return {
                "success": False,
                "error": "Failed to generate any solutions",
                "final_code": "",
                "reasoning": "",
                "confidence": 0.0
            }

        # Step 3: Aggregate solutions using majority vote
        print("🗳️  Aggregating solutions using majority vote...")
        aggregated = self.aggregate_solutions_majority_vote(solutions)

        result = {
            "success": True,
            "final_code": aggregated["code"],
            "reasoning": aggregated["reasoning"],
            "confidence": aggregated["confidence"],
            "num_solutions_generated": len(solutions),
            # "vote_distribution": aggregated["vote_distribution"],
            "all_solutions": solutions
        }

        print(f"✅ Self-consistency complete! Confidence: {aggregated['confidence']:.2f}")
        return result
