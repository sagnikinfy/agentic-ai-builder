def init_prompt(inp: str) -> str:
    return f"""You are a quality analyst of the case #{inp} . Please go through all the details of the case including the Support and the Customer's conversation messages and analyze below mentioned key metrics. Try to generate a concise and clear answer.\n\n"""

q_issue_details = """
         - Create an analysis about whether the Support correctly understood the issue and gathered sufficient information. 
        - Create an analysis whether the Support navigated effectively through internal tools and support articles to investigate the issue.
        - Create an analysis whether the Support failed to replicate or reproduce the Customer's issue to diagnose the issue.\n\n"""

q_knowledge = """
        - Create an analysis whether incorrect/ incomplete/ Workaround provided by the Support.
        - Create an analysis whether support has failed to provide step-by-step guidance while troubleshooting.
        - Create an analysis whether Customer feels that the Support was not aware of product specific concepts or the troubleshooting was not done properly.\n\n"""

q_transfer = """
        - Analyse whether Customer got unhappy or frustrated due to multiple Support agents worked on the case. Do not assume anything which is not explicitly mentioned the given context.\n\n"""


q_delay_1 = """
        - Check whether TRT and IRT SLO met or not in the case.\n"""
q_delay_2 = """
        - FMR SLA is 8 hours. From the SLA details analyze if FMR was sent before 8 hours time in the case.\n"""
q_delay_3 = """
        - From the SLA details analyze if all the INCRUPs were sent before 1 day time in the case."""
q_delay_4 = """
        - Document any incident when the Support promised the Customer to get back to them with further updates within a timeline but failed to update within that time. That made the Customer eagerly ask for updates in their next message after the promised time is over.\n\n"""


q_expectation = """
        - Check whether Support did not informed the Customer about a potential transfer, bug, consult which led to confusion. 
        - Analyze whether Support did not set clear expectations on product-specific concepts like feature requests, subscription limitations, Scope of support limitations and Propagation timelines to the Customer.
        - Check whether Supoprt closed the case without Customer communication leading to Customer came back and reopen the case.\n\n"""


q_comm = """
        - From Support messages analyze whether Support messages were poorly written and lack of good communication skills.
        - From Support messages analyze whether Support used informal tone, used jargons or wasn't polite during the conversation.
        - From Support messages analyze whether Support messages poorly displayed empathey to the Customer.
        - Analyze if the Support could not answer promptly when customer asks for information.
        - Check whether Support is reluctant to say 'no' if out of scope.
        
        Please focus on Support and Customer reply messages only. Do not consider anything from internal notes.\n\n"""


q_cont = """
        - From the Customer and Support messages analyze whether Customer was not happy with the Support channel (Inclusive with issues on PIN generation on phone support) or not.
        - Analyze if the Customer was not happy with the Support language / country with restricted access / coverage for Customer's timezone.\n\n"""

q_billing = """
        - From the Customer messages analyze if the Customer was unhappy with payment / refunds / discounts and license policies in this case.
        - From the Customer messages analyze if the Customer was unable to change billing preferences.\n\n"""

q_privacy = """
        - From the Customer messages analyze if Customer was unhappy with Data privacy Policies (permissions & access issues) in this case.
        - From the Customer messages check if Customer's account/services were interrupted due to TOS (Terms of service) violations.
        - From the Customer messages check if Customer requests the TOS (Terms of service) violation policies to be more clear and understandable.\n\n"""

q_bug = """
        - Check if there is any BUG or feature request associated with this case and that Bug or feature request took a long time to resolve.\n\n"""


q_internal = """
        - Analyze whether the Customer received delayed response from internal teams, which made the Customer unhappy.
        - Analyze if the Customer felt that response from internal team was incorrect or lacked detail.\n\n"""


q_troubleshoot = """
        - Analyze whether the troubleshooting process by the Support was lengthy and complex and that forced the Customer to self-resolve their issues.\n\n"""


q_tech = """
        - From the Customer and Support messages, analyze whether Customer was unable to implement the solution provided by Support. Also check whether Support clearly explained the solution or not.\n\n"""

q_end = "Provide supporting evidence with your analysis."

q_sentiment = """
        Analyse the overall sentiment of the Customer, whether it is positive or negative.
        - positive sentiment: Customer is satisfied with the support quality as the Support agent resolved the Customer's issues without any delay.
        - negative sentiment: Customer is unhappy with the support quality. The Support agent takes long time to resolve issue or could not understand the Customer's issues properly or failed to show emapathy.\n\n"""

q_summary = """
        Generate an overall summary of the case, including:
        - The issue that the Customer faced.
        - Root cause of the issue.
        - What all steps taken by the Support to resolve the issue.
        - The solution provided by the Support. Mention any important urls or links provided by the Support.
        - What all next steps for the Customer to do as directed by the Support.\n\n"""

q_opportunity = """- Analyse the areas to improve for the agent or TSR to improve the quality of support.\n\n"""
q_status_chng = """- Display all the case status change history, with the timeline in a tabular format.\n\n"""
q_priority_chng = """- Display all the case priority change history, with the timeline in a tabular format.\n\n"""
q_esc_reason = """- Analyse the key reaons for which the case got escalated.\n\n"""
q_dsat_reason = """- Analyse what are the key reasons for which the Support agent got low rating from the Customer.\n\n"""
            