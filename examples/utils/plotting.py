import numpy as np

def required_img_scale(min_dpi: int, target_width_cm: float, base_width_px: int, aspect_ratio: list[int]) -> tuple[int|float]:
	min_img_scale = (min_dpi * target_width_cm) / (base_width_px * aspect_ratio[0] * 2.54)
	img_scale = int(np.ceil(min_img_scale))
	new_dpi = (base_width_px * aspect_ratio[0] * img_scale * 2.54)/target_width_cm

	return img_scale, new_dpi

def required_font_size(target_pt: int, target_width_cm: float, aspect_ratio: list[int], window_size: list[int]) -> tuple[int]:
	size_cm = target_pt/72 * 2.54
	target_height_cm = target_width_cm/aspect_ratio[0] * aspect_ratio[1]
	fac = size_cm/target_height_cm
	label_size = int(np.ceil(window_size[1]*fac))
	title_size = int(np.ceil(window_size[1]*fac*1.2))

	return label_size, title_size