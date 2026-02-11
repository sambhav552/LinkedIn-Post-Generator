import json
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.exceptions import OutputParserException
from llm_helper import llm


def process_posts(raw_file_path,processed_file_path = "data/processed_data.json"):
    enriched_post = []
    with open(raw_file_path, encoding="utf-8") as file:
        posts = json.load(file)
        for post in posts:
            metadata = extract_data(post['text'])
            post_with_metadata = post | metadata 
            enriched_post.append(post_with_metadata)
    
    unified_tags = get_unified_tags(enriched_post)

    for post in enriched_post:
        current_tags = post["tags"]
        new_tags = {unified_tags[tag] for tag in current_tags}
        post["tags"] = list(new_tags)


    print("enriched posts----------",enriched_post)
    
    with open(processed_file_path,encoding = 'utf-8', mode = "w") as file:
        json.dump(enriched_post,file, indent = 4)


def get_unified_tags(post_with_metdata):
    unique_tags = set()
    for post in post_with_metdata:
        unique_tags.update(post['tags'])

    unique_tags_list = ', '.join(unique_tags)

    template = '''I will give you a list of tags. You need to unify tags with the following requirements,
    1. Tags are unified and merged to create a shorter list. 
       Example 1: "Jobseekers", "Job Hunting" can be all merged into a single tag "Job Search". 
       Example 2: "Motivation", "Inspiration", "Drive" can be mapped to "Motivation"
       Example 3: "Personal Growth", "Personal Development", "Self Improvement" can be mapped to "Self Improvement"
       Example 4: "Scam Alert", "Job Scam" etc. can be mapped to "Scams"
    2. Each tag should be follow title case convention. example: "Motivation", "Job Search"
    3. Output should be a JSON object, No preamble
    3. Output should have mapping of original tag and the unified tag. 
       For example: {{"Jobseekers": "Job Search",  "Job Hunting": "Job Search", "Motivation": "Motivation}}
    
    Here is the list of tags: 
    {tags}
    '''

    temp = PromptTemplate.from_template(template)
    chain = temp | llm
    response = chain.invoke(input={'tags':str(unique_tags_list)})
    try:
        json_parser = JsonOutputParser()
        res = json_parser.parse(response.content)
    except OutputParserException: 
        raise OutputParserException("Context too big. Unable to parse jobs.")
    return res




def extract_data(post):
    template = '''
    You are given a LinkedIn post. You need to extract number of lines, language of the post and tags.
    1. Return a valid JSON. No preamble. 
    2. JSON object should have exactly three keys: line_count, language and tags. 
    3. tags is an array of text tags. Extract maximum two tags.
    4. Language should be English or Hinglish (Hinglish means hindi + english)
    
    Here is the actual post on which you need to perform this task:  
    {post}
    '''
    ppt = PromptTemplate.from_template(template)
    chain = ppt | llm
    response = chain.invoke(input = {'post':post})

    try:
        json_parser = JsonOutputParser()
        res = json_parser.parse(response.content)
    except OutputParserException: 
        raise OutputParserException("Context too big. Unable to parse jobs.")
    return res
    


if __name__ == "__main__":
    process_posts("data/raw_posts.json","data/preprocessed_data.json")