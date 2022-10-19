from extract_descriptors import extract_descriptors

subdir = 'train'
num_rand=3

associations_dir = '/home/harry/harry_ws/fruitlet/fruitlet_assoc_train/data/fruitlet_assoc_annotations/' + subdir
output_dir = '/home/harry/harry_ws/fruitlet/fruitlet_assoc_train/AssociationNetwork/datasets/' + subdir

tag_segment_dir = '/home/harry/harry_ws/fruitlet/bbox_assoc_train/data/tag_mask_annotations/full'
box_segment_dir = '/home/harry/harry_ws/fruitlet/bbox_assoc_train/data/fruitlet_mask_annotations/full'
disparity_dir = '/home/harry/harry_ws/fruitlet/bbox_assoc_train/data/docker_extract/disparities'
model_path = '/home/harry/harry_ws/fruitlet/bbox_assoc_train/data/mix_model.pth'

extract_descriptors(associations_dir, tag_segment_dir, box_segment_dir, disparity_dir, model_path, output_dir, num_rand=num_rand)
