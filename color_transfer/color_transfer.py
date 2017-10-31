def get_img_cov(img):
	return np.cov(img.reshape((-1, 3)).T)


def transfer_color(src, tgt):
	res = np.zeros_like(tgt)

	src = src.astype(np.float32, copy=True)
	tgt = tgt.astype(np.float32, copy=True)

	src /= 255.0
	tgt /= 255.0

	src_cov = get_img_cov(src)
	tgt_cov = get_img_cov(tgt)

	U_src, A_src, V_src = np.linalg.svd(src_cov)
	U_tgt, A_tgt, V_tgt = np.linalg.svd(tgt_cov)

	src_mean_values = np.mean(src, axis=(0,1))
	tgt_mean_values = np.mean(tgt, axis=(0,1))

	R_src, R_tgt = U_src, U_tgt
	R_src = np.vstack((np.hstack((R_src,[[0],[0],[0]])), [0,0,0,1]))
	R_tgt = np.vstack((np.hstack((R_tgt,[[0],[0],[0]])), [0,0,0,1]))
	R_tgt = np.linalg.inv(R_tgt)

	t_src_r, t_src_g, t_src_b = src_mean_values
	t_tgt_r, t_tgt_g, t_tgt_b = -tgt_mean_values

	T_src = np.array([[1.0, 0, 0, t_src_r], [0, 1.0, 0, t_src_g], [0, 0, 1.0, t_src_b], [0, 0, 0, 1]])
	T_tgt = np.array([[1.0, 0, 0, t_tgt_r], [0, 1.0, 0, t_tgt_g], [0, 0, 1.0, t_tgt_b], [0, 0, 0, 1]])

	s_src_r, s_src_g, s_src_b = math.sqrt(A_src[0]), math.sqrt(A_src[1]), math.sqrt(A_src[2])
	s_tgt_r, s_tgt_g, s_tgt_b = 1.0 / math.sqrt(A_tgt[0]), 1.0 / math.sqrt(A_tgt[1]), 1.0 / math.sqrt(A_tgt[2])
	S_src = np.array([[s_src_r, 0.0, 0, 0], [0, s_src_g, 0, 0.0], [0, 0, s_src_b, 0.0], [0, 0, 0, 1]])
	S_tgt = np.array([[s_tgt_r, 0.0, 0, 0], [0, s_tgt_g, 0, 0.0], [0, 0, s_tgt_b, 0.0], [0, 0, 0, 1]])

	dot_product = reduce(np.dot, [T_src, R_src, S_src, S_tgt, R_tgt, T_tgt])

	res = np.insert(tgt, 3, 1, axis=2)
	res = np.reshape(res, (-1, 4))
	# res = np.dot(res, dot_product)
	res = np.dot(dot_product, res.T).T
	res = np.reshape(res, (tgt.shape[0], tgt.shape[1], 4))
	res = np.clip(255 * res[:,:,:-1], 0, 255)
	res = res.astype(np.uint8)

	return res


import cv2, math
import numpy as np





src_img = cv2.imread(R'C:\Users\cysiek\AMD APP SDK\3.0\samples\opencl\cpp_cl\1.x\SobelFilterImage2\SobelFilterImage_Input.bmp')
tgt_img = cv2.imread(R'C:\Users\cysiek\AMD APP SDK\3.0\samples\opencl\cpp_cl\1.x\SobelFilterImage2\lena_SobelFilterImage_Input.bmp')


res = transfer_color(src_img, tgt_img)
cv2.imwrite(R'C:\Users\cysiek\AMD APP SDK\3.0\samples\opencl\cpp_cl\1.x\SobelFilterImage2\res.bmp', res)

