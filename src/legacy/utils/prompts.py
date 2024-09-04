DEFAULT_CITATION_EXTRACTION_PROMPT_TEMPLATE = """
You are about to analyze segments of academic texts where authors cite other papers. 

### Instruction
Your task is to identify and articulate the nature of the relationship between the citing paper and the cited works. Focus on understanding why the author has chosen to reference these works and in what context. Use the following guidelines to structure your response:

- Identify the Citation: Start by locating the citation within the text. Citations might appear in various formats, such as numbers in brackets, author names, or footnotes.

- Contextual Clues: Read the sentences or paragraphs surrounding the citation to gather context. Pay attention to verbs and phrases that might indicate the relationship, such as "builds upon," "contradicts," "extends," "employs," or "reviews."

- Classify the Relationship: Based on the context and the way the cited work is discussed, classify the relationship into one of the following categories:

    - Supporting Evidence: The cited work provides foundational data, theories, or methodologies that support the claims or hypotheses of the citing paper.
    - Methodological Basis: The citing paper adopts or adapts the methods, techniques, or procedures from the cited work.
    - Theoretical Foundation: The citing paper leverages theories, models, or frameworks established in the cited work to underpin its own research.
    - Data Source: The citation is used to acknowledge the origin of a dataset, model, or specific information that the citing paper utilizes in its research or analysis. This category highlights the reliance on external data or pre-existing models as a foundational element for the study conducted in the citing paper.
    - Extension or Continuation: The citing paper expands upon the research of the cited work, exploring new dimensions, contexts, or variables.

For each citation, provide a brief summary in the format of JSON that includes:

- The citation number (as presented in the text).
- The category of the relationship (from the classifications provided above).
- A short explanation of how the cited work contributes to the citing paper.
The format of the output should be strictly in JSON format in the form: 
    
{{
    "Citation": "[citation number]",
    "Category": "[relationship category]",
    "Explanation": "[brief explanation of the relationship]"
}}

Example1 :
{{
    "Citation": "[1]",
    "Category": "Supporting Evidence",
    "Explanation": "The cited work provides a foundational algorithm that the citing paper leverages to improve computational efficiency in data analysis."
}}
Example 2:
{{
    "Citation": "(Bai et al. 2018)",
    "Category": "Supporting Evidence",
    "Explanation": "The cited work presents a comprehensive review of the literature on the topic, providing a strong theoretical foundation for the claims made in the citing paper."
}}

No explanation is needed
### Input
{input}
### Response
"""

DEFAULT_CITATION_INFER_PROMPT_TEMPLATE = """You are about to analyze segments of academic texts where authors cite other papers. 

### Instruction
Your task is to identify and articulate the nature of the relationship between the citing paper and the cited works. Focus on understanding why the author has chosen to reference these works and in what context. Use the following guidelines to structure your response:

- Identify the Citation: Start by locating the citation within the text. Citations might appear in various formats, such as numbers in brackets, author names, or footnotes.

- Contextual Clues: Read the sentences or paragraphs surrounding the citation to gather context. Pay attention to verbs and phrases that might indicate the relationship, such as "builds upon," "contradicts," "extends," "employs," or "reviews."

- Classify the Relationship: Based on the context and the way the cited work is discussed, classify the relationship into one of the following categories:

    - Supporting Evidence: The cited work provides foundational data, theories, or methodologies that support the claims or hypotheses of the citing paper.
    - Methodological Basis: The citing paper adopts or adapts the methods, techniques, or procedures from the cited work.
    - Theoretical Foundation: The citing paper leverages theories, models, or frameworks established in the cited work to underpin its own research.
    - Data Source: The citation is used to acknowledge the origin of a dataset, model, or specific information that the citing paper utilizes in its research or analysis. This category highlights the reliance on external data or pre-existing models as a foundational element for the study conducted in the citing paper.
    - Extension or Continuation: The citing paper expands upon the research of the cited work, exploring new dimensions, contexts, or variables.

For each citation, provide a brief summary in the format of JSON that includes:

- The citation number (as presented in the text).
- The category of the relationship (from the classifications provided above).
- A short explanation of how the cited work contributes to the citing paper.

### Input:
{input}

### Response:
"""