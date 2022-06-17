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


def get_all_conversation_raw_data_tsv(conversation_list) -> list:
    posts = conversation_list
    data = [extract_post_raw_data(post, index) for index, post in enumerate(posts)]
    return data