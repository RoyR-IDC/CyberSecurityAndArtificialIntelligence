import xml.etree.ElementTree
import xml.etree.ElementTree as ET


def get_tree():
    path = r'/Users/royrubin/PycharmProjects/CyberSecurityAndArtificialIntelligence/FinalProject/CleanedFormattedData/ArmySgt1961.xml'
    tree = ET.parse(path)

    # print root info
    root = tree.getroot()
    print(f'root {root} root.tag {root.tag} root.attrib {root.attrib}')

    return tree


def print_all_tags(tree):
    elem_tags_list = []

    for elem in tree.iter():
        elem_tags_list.append(elem.tag)

    # now I remove duplicities - by convertion to set and back to list
    elem_tags_list = list(set(elem_tags_list))

    # Just printing out the result
    print(elem_tags_list)


def extract_post_raw_data(post_elem: xml.etree.ElementTree.Element):
    """
    post example:

    <POST>
    <USERNAME>armysgt1961</USERNAME>
    <DATETIME>(7:02:01 pm)</DATETIME>
    <BODY>im dennis us army soldier from cincinnati</BODY>
    <COMMENT></COMMENT>
    <CODING>
        <VERSION>ArmySgt1961.elkilmer.coded.txt</VERSION>
        <CATEGORY>200</CATEGORY>
        <TONE>1</TONE>
    </CODING>
    </POST>

    :param post_elem:
    :return:
    """
    output = {}

    # init
    output['id'] = None
    output['datetime'] = None
    output['talk'] = None
    output['comment'] = None
    output['category'] = None
    output['tone'] = None

    if post_elem is None:
        return output  # all Nones

    # Step 1:
    output['id'] = post_elem.find("USERNAME").text if post_elem.find("USERNAME") is not None else None
    output['datetime'] = post_elem.find("DATETIME").text if post_elem.find("DATETIME") is not None else None
    output['talk'] = post_elem.find("BODY").text if post_elem.find("BODY") is not None else None
    output['comment'] = post_elem.find("COMMENT").text if post_elem.find("COMMENT") is not None else None

    # Step 2:
    coding = post_elem.find("CODING")
    if coding is None:
        return output  # all Nones

    output['category'] = coding.find("CATEGORY").text if coding.find("CATEGORY") is not None else None
    output['tone'] = coding.find("TONE").text if coding.find("TONE") is not None else None

    # Step 3: return if everything worked
    return output


def get_all_conversation_raw_data(tree):
    posts = list(tree.iter("POST"))  # equivalent: posts = [elem for elem in tree.iter() if elem.tag == 'POST']
    data = [extract_post_raw_data(post) for post in posts]
    return data


def generate_features_from_conversation(file_name):
    features = {}
    features['file_id'] = file_name


def process_single_file():
    pass
    # TODO
    generate tree from file
    get the data
    generate features

def generate_dataframe_from_data():
    pass
    # TODO
    for file in files:
        open file
        process_single_file()
        add to dataframe



def main():
    tree = get_tree()
    print_all_tags(tree)
    # print_all_conversations(tree)
    data = get_all_conversation_raw_data(tree)
    print(data[0])


if __name__ == '__main__':
    main()
