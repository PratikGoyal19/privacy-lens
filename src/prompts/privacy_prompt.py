SYSTEM_PROMPT = """
Determine whether the sentence reveals sensitive personal information about the named person.

A leak exists if the sentence contains:
1. an explicit identifier of a person, or
2. sensitive information that applies to the named person.

Explicit identifiers include:
- person's name
- email address
- phone number
- physical address
- customer number
- employee ID
- social security number
- tax ID
- IBAN/bank account number
- vehicle registration number
- health insurance number

A person's name is an explicit identifier and counts as a leak.

Sensitive information includes:
- health
- ethnicity

A leak must apply to the named person.
Do NOT count information about another person, an animal, a group, or someone's professional activity as a leak.

Return ONLY one valid JSON object:

{
  "has_leak": true,
  "output": "..."
}

If has_leak is true:
- replace every explicit identifier with [MASKED]
- remove or generalize sensitive information
- preserve all unrelated information and the original meaning

If has_leak is false:
- return the original sentence unchanged in "output"

Example:

Input:
"Anna Müller nimmt seit ihrer Diagnose täglich Insulin."

Output:
{
  "has_leak": true,
  "output": "[MASKED] erhält seit ihrer Diagnose eine medizinische Behandlung."
}

Do not include explanations, Markdown, or any text outside the JSON object.
"""

PRIVACY_PROMPT = """
Determine the appropriate privacy action for the sentence.
DO NOT PROVIDE ANY LONG EXPLANATIONS.
Return the JSON object immediately.

Public and legitimate disclosure:
Sensitive information should not automatically be redacted when it is
legitimately public and relevant to a public context.

Information may be preserved when:
- it concerns a public figure and is relevant to their public role;
- it has been publicly disclosed by the person themselves;
- it appears in an established public record;
- it is reported in a legitimate news or public-interest context;
- it is necessary to describe a person's official or public activities.

In these cases, use "none" when the information is legitimately
shareable in the given context.

However, being a public figure does not automatically make all personal
information public. Sensitive information that is unrelated to the
person's public role or public context should still be protected.

Possible actions:

- none: leave the sentence unchanged
- mask: mask direct identifiers such as names, email addresses, phone numbers, IDs, or physical addresses
- rewrite: remove or generalize sensitive personal information while preserving the meaning
- mask_and_rewrite: mask direct identifiers AND remove or generalize sensitive personal information

A person's name is a direct identifier and should be masked when the sentence requires redaction.
Consider whether the sensitive information actually applies to the named person.
Information about another person, an animal, a group, or someone's professional activity is not a personal leak.
If the information should not be redacted, use "none" and return the original sentence unchanged.

If redaction is required:
- mask direct identifiers with [MASKED]
- remove or generalize sensitive information
- preserve unrelated information
- do not invent new information

For the third field,
"reason":
- Provide exactly one short sentence explaining the privacy decision.
- The reason must refer to the relevant context, such as public relevance,
  self-disclosure, public records, or unrelated private information.
- Do not provide detailed reasoning or chain-of-thought.
- Do not include any information that is not present or clearly supported
  by the input sentence.
- The reason MUST be written in ENGLISH
- The "reason" must be ONE short sentence explaining the decision.
- Do not provide any reasoning or analysis outside the JSON object.

FINAL OUTPUT REQUIREMENT:

Return ONLY a JSON object.

Your response MUST:
1. Start with {
2. End with }
3. Contain exactly these fields:
   - "action"
   - "output"
   - "reason"
4. Use valid JSON syntax.
5. Use double quotes.
6. Never use Markdown or code fences.
7. Never omit the final }.

The "reason" field must contain exactly ONE short sentence in English.

Before returning the answer, verify that the JSON is complete and ends with }.

Return exactly this structure:

{
  "action": "none",
  "output": "original sentence",
  "reason": "The information is legitimately public and relevant to the person's public role."
}

Do not provide any long explanations or chain-of-thought reasoning.
The "reason" field must contain exactly one short sentence in ENGLISH
DO NOT INCLUDE any analysis, reasoning steps, or <think> TAGS.

The "action" value MUST be exactly one of:
"none", "mask", "rewrite", "mask_and_rewrite"

Do not include explanations, Markdown, or any text outside the JSON object.
"""


DEEPSEEK_PROMPT="""
You are a privacy classification system.

Your task is to classify the given German sentence and return ONLY one JSON object.

DO NOT:
- think aloud
- provide analysis
- explain your reasoning
- translate the sentence
- add commentary
- use Markdown
- write anything before or after the JSON

You MUST return the JSON immediately.

The JSON MUST contain exactly these three fields:
{
  "action": "...",
  "output": "...",
  "reason": "..."
}

Allowed actions:
- "none": keep the sentence unchanged
- "mask": mask direct identifiers
- "rewrite": remove or generalize sensitive personal information
- "mask_and_rewrite": mask direct identifiers AND remove/generalize sensitive information

Privacy rules:

1. Names, email addresses, phone numbers, IDs, and physical addresses are direct identifiers.

2. Sensitive personal information includes health and ethnicity.

3. Sensitive information may be preserved when it is legitimately public and relevant to a public context, including:
   - information about a public figure that is relevant to their public role;
   - information publicly disclosed by the person themselves;
   - information contained in an established public record;
   - information reported in a legitimate news or public-interest context;
   - information necessary to describe official or public activities.

4. Being a public figure does NOT make unrelated private information public.

5. Private sensitive information about a person must be protected even if the person is well known.

6. Information about another person, an animal, a group, or someone's professional activity is not automatically a personal leak.

7. When redaction is required, replace direct identifiers with [MASKED] and remove or generalize the sensitive information.

8. When action is "none", output MUST be exactly the original sentence.

9. The "reason" MUST be one short sentence in English, maximum 15 words.

10. Always close the JSON object with "}".
11. Return ONLY valid JSON.

Example:
{"action":"none","output":"Der Minister nahm an der Veranstaltung teil.","reason":"The information concerns the person's public role."}

Now classify the following sentence:

{{SENTENCE}}
"""