def extract_post_raw_data(talk_string: str, index: int):
    # prep
    output = {}

    # init
    output['id'] = 'person1' if index % 2 != 0 else 'person2'
    output['datetime'] = None
    output['talk'] = talk_string
    output['comment'] = None
    output['category'] = None
    output['tone'] = None

    return output


def get_all_conversation_raw_data_yml(conversation_lists) -> list:
    flat_list = [x for inner in conversation_lists for x in inner]
    posts = flat_list
    data = [extract_post_raw_data(post, index) for index, post in enumerate(posts)]
    return data