from collections import defaultdict
import re
import json
import ast

def has_citation(text):
  """Checks if a text string has citation marks.

  Args:
    text: The text string to check.

  Returns:
    True if the text has citation marks, False otherwise.
  """

  # Regex patterns for different citation formats
  patterns = [
      r"\[[\d,]+\]",  # for citations like [1], [1, 3, 4]
      r"\s*et al\.\s*",  # for citations like (Author et al. 2020)
  ]

  for pattern in patterns:
    if re.search(pattern, text):
      return True
  return False


def parse_json(string):
    pattern = r'{(.*?)}'

    # Find all matches of the pattern in the original string
    matches = re.findall(pattern, string, re.DOTALL)

    # List to store extracted JSON objects
    json_objects = []

    # Iterate over matches and append them to the list
    for match in matches:
        match = "{" + match.strip() + "}"
        try:
            # match = ast.literal_eval(match)
            match = match.replace("': '", '": "').replace("', '", '", "').replace("'}", '"}').replace("{'", '{"').replace("':", '":')        
            json_objects.append(json.loads(match))
        except Exception as e:
            print(f"Error parsing JSON for chunk: {match}")
            continue

    return json_objects

def split_and_group(data):
    grouped_data = defaultdict(lambda: {'Categories': set(), 'Explanation': ''})
    
    for item in data:
        # Extract citation numbers from the string, assuming they are integers
        citations = [int(c) for c in item['Citation'].strip('[]').split(', ')]
        
        for citation in citations:
            grouped_data[citation]['Categories'].add(item['Category'])
            grouped_data[citation]['Explanation'] += item['Explanation'] + ' '
    
    # Convert the set of categories to a list and trim trailing spaces from explanations
    for citation, details in grouped_data.items():
        details['Categories'] = list(details['Categories'])
        details['Explanation'] = details['Explanation'].strip()
    
    return grouped_data

