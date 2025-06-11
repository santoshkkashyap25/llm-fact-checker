# # test claim_extractor
# from core.claim_extractor import claim_extractor

# # Sample inputs
# examples = [
#     "The company launched a new product last week.",
#     "Climate change is a serious threat to humanity.",
#     "Apple announced its earnings report.",
#     "There was a loud noise in the street.",
#     "Innovation drives progress."
# ]

# # Run extraction
# for i, text in enumerate(examples, 1):
#     claim = claim_extractor.extract(text)
#     print(f"Example {i}:")
#     print(f"Input: {text}")
#     print(f"Extracted Claim: {claim}")
#     print()


# # test llm_service
# from core.llm_service import llm_service

# # Sample claim and evidence
# claim = "The Eiffel Tower is located in Berlin."
# evidence = [
#     "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.",
#     "It was named after engineer Gustave Eiffel, whose company designed and built the tower."
# ]

# # Get verdict
# verdict = llm_service.get_verdict(claim, evidence)

# # Print result
# print("Verdict:", verdict.verdict)
# print("Confidence:", verdict.confidence)
# print("Reasoning:", verdict.reasoning)











# test pipeline

from pipeline import run_fact_checking_pipeline

if __name__ == "__main__":
    sample_text = "The Ayushman Bharat Pradhan Mantri Jan Arogya Yojana provides health insurance coverage of up to ₹5 lakh per family per year."
    
    try:
        result = run_fact_checking_pipeline(sample_text)
        print("Pipeline Output:")
        for key, value in result.items():
            print(f"{key}: {value}")
    except Exception as e:
        print(f"Error occurred: {e}")
