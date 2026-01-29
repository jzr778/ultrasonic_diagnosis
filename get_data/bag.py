from get_meta_data import get_meta_data

def get_bag(tag_id_list):
    for tag_id in tag_id_list:
        meta_data = get_meta_data(tag_id=tag_id)
        bag_name_list = meta_data['body'][0]['bagsName']
        heavy_bags = sorted([bag_name for bag_name in bag_name_list if 'Heavy' in bag_name])
        for bag in heavy_bags:
            print(bag)

if __name__ == '__main__':
    tag_id_list = [
        # 98724502, 98679616, 98502291, 98392892, 98383844, 98321419, 98099917, 98034789, 98196500, 98081681, 97938019, 97937879
        # 99624582, 99622577, 99618544, 99615631, 99611943, 99571260, 99566777, 99552470, 99477298, 99475457, 99460115, 99458486
        # 99940155, 99907365, 99768348, 99736577, 99853011, 99722714, 99455627, 99440656, 99438271
        99997866
    ]
    get_bag(tag_id_list)
