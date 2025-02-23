import io
import sys

import os
import cv2
import numpy as np

import psutil
import math
import random
import copy
import time
import pickle
from scipy.signal import correlate, convolve
class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, lr=0.01):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.lr = lr  #Thêm dòng này để lưu learning rate

        # Xavier Initialization
        limit = np.sqrt(6 / (in_channels + out_channels))
        self.kernels = np.random.uniform(-limit, limit, 
                                         (out_channels, in_channels, kernel_size, kernel_size))

    def forward(self, input_tensor, training=False):
        print(f"[DEBUG] Conv2D Forward: Input shape {input_tensor.shape}")

        self.input = input_tensor  # Lưu đầu vào để backward
        batch_size, in_channels, height, width = input_tensor.shape

        output_height = (height - self.kernel_size + 2 * self.padding) // self.stride + 1
        output_width = (width - self.kernel_size + 2 * self.padding) // self.stride + 1

        output = np.zeros((batch_size, self.out_channels, output_height, output_width), dtype=np.float32)
        print(f"[DEBUG] Conv2D Output shape {output.shape}")

        if self.padding > 0:
            input_tensor = np.pad(input_tensor,
                                ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                                mode='constant')
            print(f"[DEBUG] Conv2D Input after padding {input_tensor.shape}")

        # Tích chập từng kernel
        for oc in range(self.out_channels):
            for ic in range(in_channels):
                kernel_reshaped = self.kernels[oc, ic][np.newaxis, :, :]
                #print(f"[DEBUG] Conv2D Kernel shape {kernel_reshaped.shape}")

                try:
                    output[:, oc] += correlate(input_tensor[:, ic], kernel_reshaped, mode='valid')
                except Exception as e:
                    print(f"[ERROR] Error while convolutional: {e}")
                    raise e

        self.output = output  # Lưu output để dùng trong backward
        print("[DEBUG] Conv2D Forward finished.")
        return output


    def backward(self, d_output):
        batch_size, in_channels, height, width = self.input.shape
        _, _, output_height, output_width = d_output.shape

        d_kernels = np.zeros_like(self.kernels)
        d_input = np.zeros_like(self.input)

        # Thêm padding nếu có
        if self.padding > 0:
            padded_input = np.pad(self.input,
                                ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                                mode='constant')
        else:
            padded_input = self.input

        # [DEBUG] Kiểm tra shape trước khi tính toán gradient
        print(f"[DEBUG] d_output shape: {d_output.shape}")
        print(f"[DEBUG] padded_input shape: {padded_input.shape}")
        print(f"[DEBUG] d_input shape: {d_input.shape}")

        # Tính gradient cho kernels (d_kernels)
        for oc in range(self.out_channels):
            for ic in range(in_channels):
                for b in range(batch_size):
                    d_kernels[oc, ic] += correlate(padded_input[b, ic], d_output[b, oc], mode='valid')

        # Tính gradient cho input (d_input)
        for ic in range(in_channels):
            for oc in range(self.out_channels):
                flipped_kernel = np.flip(self.kernels[oc, ic], axis=(0, 1))
                for b in range(batch_size):
                    conv_result = convolve(d_output[b, oc], flipped_kernel, mode='full')

                    # Đảm bảo gradient có cùng shape với `d_input`
                    expected_shape = d_input.shape[2:]  # (height, width)
                    crop_h = (conv_result.shape[0] - expected_shape[0]) // 2
                    crop_w = (conv_result.shape[1] - expected_shape[1]) // 2

                    if crop_h > 0 or crop_w > 0:
                        conv_result = conv_result[crop_h:-crop_h, crop_w:-crop_w]

                    d_input[b, ic] += conv_result

        # Cập nhật trọng số (Gradient Descent)
        self.kernels -= self.lr * d_kernels

        # Fix lỗi: Lưu gradient để các layer sau sử dụng
        self.dinputs = d_input  

        # [DEBUG] Kiểm tra shape sau khi backward
        print(f"[DEBUG] d_kernels shape: {d_kernels.shape}")
        print(f"[DEBUG] d_input shape: {d_input.shape}")

        return d_input

class Linear:
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features
        self.weights = np.random.uniform(-1, 1, (in_features, out_features))
        self.biases = np.random.uniform(-1, 1, (1, out_features))
        self.dweights = np.zeros_like(self.weights)
        self.dbiases = np.zeros_like(self.biases)

    def forward(self, input_tensor, training=False):  # Thêm training=False
        """
        Tính toán forward của lớp Linear (Fully Connected).
        :param input_tensor: NumPy array có shape (batch_size, in_features)
        :param training: Cờ kiểm tra đang training hay inference (không sử dụng ở đây)
        :return: NumPy array có shape (batch_size, out_features)
        """
        self.input_tensor = input_tensor  # Lưu input để dùng trong backward
        self.output = np.dot(input_tensor, self.weights) + self.biases
        return self.output

    def backward(self, d_loss):
        """
        Truyền ngược gradient qua lớp Linear.
        :param d_loss: Gradient của loss đối với output của Linear
        :return: Gradient của loss đối với input của Linear
        """
        print(f"[DEBUG] Linear backward input shape: {d_loss.shape}")  # 🛠️ Thêm debug

        self.dweights = np.dot(self.input_tensor.T, d_loss)
        self.dbiases = np.sum(d_loss, axis=0, keepdims=True)
        
        self.dinputs = np.dot(d_loss, self.weights.T)  
        print(f"[DEBUG] Linear dinputs shape: {self.dinputs.shape}")

        if d_loss.shape[1] != self.weights.shape[1]:
            print(f"[WARNING] d_loss shape {d_loss.shape} not match, need reshape...")
            d_loss = d_loss.reshape(-1, self.weights.shape[1])  # Thử reshape

        # Kiểm tra kích thước trước khi nhân ma trận
        assert self.weights.shape[1] == d_loss.shape[1], \
            f"[ERROR] Shape mismatch: {self.weights.shape} vs {d_loss.shape}"

        return np.dot(d_loss, self.weights.T)  # Trả về gradient để truyền ngược


class Sigmoid:
    def forward(self, input_tensor, training=False):
        epsilon = 1e-9  # 🔥 Tránh mất gradient
        self.output = 1 / (1 + np.exp(-np.clip(input_tensor, -500, 500)))
        self.output = np.clip(self.output, epsilon, 1 - epsilon)  # 🔥 Thêm clipping
        print(f"[DEBUG] Sigmoid output (min: {self.output.min()}, max: {self.output.max()})")
        return self.output

    def backward(self, d_loss):
        """
        Truyền ngược gradient qua Sigmoid.
        """
        self.dinputs = d_loss * (self.output * (1 - self.output))  # 🔥 Fix lỗi `next`
        return self.dinputs

    def predictions(self, output_tensor):
        """
        Chuyển đầu ra thành nhãn nhị phân (0 hoặc 1).
        """
        return output_tensor > 0.5  # Nếu > 0.5, trả về 1; ngược lại, trả về 0

class BatchNorm1D:
    def __init__(self, num_features, momentum=0.9, epsilon=1e-5):
        self.num_features = num_features
        self.momentum = momentum
        self.epsilon = epsilon

        self.gamma = np.ones((1, num_features))
        self.beta = np.zeros((1, num_features))

        self.running_mean = np.zeros((1, num_features))
        self.running_var = np.ones((1, num_features))

    def forward(self, input_tensor, training=True):
        self.input = input_tensor  # 🔥 Lưu input để dùng trong backward()
        
        if training:
            self.batch_mean = np.mean(input_tensor, axis=0, keepdims=True)
            self.batch_var = np.var(input_tensor, axis=0, keepdims=True)

            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self.batch_var

            self.x_hat = (input_tensor - self.batch_mean) / np.sqrt(self.batch_var + self.epsilon)
        else:
            self.x_hat = (input_tensor - self.running_mean) / np.sqrt(self.running_var + self.epsilon)

        self.output = self.gamma * self.x_hat + self.beta
        return self.output

    def backward(self, d_loss):
        batch_size = d_loss.shape[0]

        d_gamma = np.sum(d_loss * self.x_hat, axis=0, keepdims=True)
        d_beta = np.sum(d_loss, axis=0, keepdims=True)

        d_x_hat = d_loss * self.gamma
        d_var = np.sum(d_x_hat * (self.input - self.batch_mean) * -0.5 * np.power(self.batch_var + self.epsilon, -1.5), axis=0, keepdims=True)
        d_mean = np.sum(d_x_hat * -1 / np.sqrt(self.batch_var + self.epsilon), axis=0, keepdims=True) + d_var * np.mean(-2 * (self.input - self.batch_mean), axis=0, keepdims=True)

        self.dinputs = d_x_hat / np.sqrt(self.batch_var + self.epsilon) + d_var * 2 * (self.input - self.batch_mean) / batch_size + d_mean / batch_size
        return self.dinputs


class ReLU:
    def forward(self, input_tensor, training=False):
        """
        Áp dụng hàm ReLU trên tensor đầu vào.
        :param input_tensor: NumPy array có shape bất kỳ
        :param training: Cờ kiểm tra đang training hay inference (không sử dụng ở đây)
        :return: NumPy array sau khi áp dụng ReLU
        """
        self.mask = input_tensor > 0  # 🔥 Lưu mask nhị phân thay vì input để tiết kiệm bộ nhớ
        self.output = np.maximum(0, input_tensor)
        return self.output

    def backward(self, d_loss):
        """
        Truyền ngược gradient qua ReLU.
        :param d_loss: Gradient của loss đối với output của ReLU
        :return: Gradient của loss đối với input của ReLU
        """
        print(f"[DEBUG] ReLU backward input shape: {d_loss.shape}")  # 🛠️ Debug

        self.dinputs = d_loss * self.mask  # 🔥 Sử dụng mask để tính gradient
        print(f"[DEBUG] ReLU dinputs shape: {self.dinputs.shape}")  # 🛠️ Debug

        return self.dinputs

class MaxPool2D:
    def __init__(self, kernel_size=2, stride=2):
        self.kernel_size = kernel_size
        self.stride = stride
    def forward(self, input_tensor, training=False):
        batch_size, channels, height, width = input_tensor.shape
        pool_height, pool_width = self.kernel_size, self.kernel_size
        stride = self.stride

        output_height = (height - pool_height) // stride + 1
        output_width = (width - pool_width) // stride + 1

        self.input_shape = input_tensor.shape
        self.output = np.zeros((batch_size, channels, output_height, output_width), dtype=np.float32)

        # 🔥 Dùng NumPy thay vì dict để lưu chỉ mục max
        self.max_indices = np.zeros((batch_size, channels, output_height, output_width, 2), dtype=np.int32)

        for b in range(batch_size):
            for c in range(channels):
                for i in range(output_height):
                    for j in range(output_width):
                        start_i = i * stride
                        start_j = j * stride
                        end_i = start_i + pool_height
                        end_j = start_j + pool_width

                        region = input_tensor[b, c, start_i:end_i, start_j:end_j]
                        max_pos = np.unravel_index(np.argmax(region), region.shape)

                        self.max_indices[b, c, i, j] = [start_i + max_pos[0], start_j + max_pos[1]]
                        self.output[b, c, i, j] = region[max_pos]

        return self.output
    
    def backward(self, d_loss):
        batch_size, channels, height, width = self.input_shape
        self.dinputs = np.zeros((batch_size, channels, height, width), dtype=np.float32)  # 🔥 Lưu `self.dinputs`

        output_height, output_width = d_loss.shape[2], d_loss.shape[3]

        for b in range(batch_size):
            for c in range(channels):
                for i in range(output_height):
                    for j in range(output_width):
                        orig_i, orig_j = self.max_indices[b, c, i, j]  # 🔥 Lấy chỉ số max từ NumPy array
                        self.dinputs[b, c, orig_i, orig_j] = d_loss[b, c, i, j]

        return self.dinputs
class AdaptiveAvgPool2D:
    def __init__(self, output_size):
        """
        Lớp Adaptive Average Pooling.
        :param output_size: Kích thước đầu ra mong muốn (H_out, W_out).
        """
        self.output_size = output_size  # (H_out, W_out)

    def forward(self, input_tensor, training=False):  # Thêm training=False để tránh lỗi
        """
        Thực hiện Adaptive Average Pooling.
        :param input_tensor: NumPy array có shape (batch_size, channels, height, width)
        :param training: Cờ kiểm tra đang training hay inference (không sử dụng ở đây)
        :return: NumPy array với shape (batch_size, channels, H_out, W_out)
        """
        self.input_shape = input_tensor.shape  # Lưu lại shape để dùng trong backward
        batch_size, channels, height, width = input_tensor.shape
        H_out, W_out = self.output_size

        self.kernel_size_h = height // H_out
        self.kernel_size_w = width // W_out
        self.stride_h = self.kernel_size_h
        self.stride_w = self.kernel_size_w

        output_tensor = np.zeros((batch_size, channels, H_out, W_out))

        for i in range(H_out):
            for j in range(W_out):
                window = input_tensor[:, :, i * self.stride_h:i * self.stride_h + self.kernel_size_h,
                                      j * self.stride_w:j * self.stride_w + self.kernel_size_w]
                output_tensor[:, :, i, j] = np.mean(window, axis=(2, 3))  # Tính trung bình nhanh hơn

        self.output = output_tensor  #**Fix lỗi: Lưu output để lớp sau có thể sử dụng**
        return output_tensor

    def backward(self, d_loss):
        batch_size, channels, height, width = self.input_shape
        self.dinputs = np.zeros((batch_size, channels, height, width))  # 🔥 Fix: Lưu `self.dinputs`

        for i in range(self.output_size[0]):
            for j in range(self.output_size[1]):
                self.dinputs[:, :, i * self.stride_h:i * self.stride_h + self.kernel_size_h,
                            j * self.stride_w:j * self.stride_w + self.kernel_size_w] += (
                    d_loss[:, :, i, j][:, :, None, None] / (self.kernel_size_h * self.kernel_size_w)
                )

        return self.dinputs  # 🔥 Fix: Trả về `self.dinputs`


class Dropout:
    def __init__(self, p=0.5):
        """
        Lớp Dropout để regularization.
        :param p: Xác suất bỏ qua một neuron (0 ≤ p < 1), mặc định là 0.5.
        """
        assert 0 <= p < 1, "p phải nằm trong khoảng [0, 1)"
        self.p = p
        self.training = True  # Mặc định là training mode
        self.mask = None  # Mask sẽ được tạo trong forward

    def forward(self, input_tensor, training=True):  # Thêm training=True
        """
        Áp dụng Dropout lên input.
        :param input_tensor: NumPy array (batch_size, features)
        :param training: Cờ kiểm tra đang training hay inference
        :return: NumPy array sau khi áp dụng Dropout
        """
        self.training = training  # Cập nhật trạng thái training

        if not self.training:
            return input_tensor  # Không dropout khi inference

        # Tạo mask (1 giữ lại, 0 bỏ đi), chia (1 - p) để giữ giá trị kì vọng
        self.mask = (np.random.rand(*input_tensor.shape) > self.p) / (1 - self.p)
        self.output = input_tensor * self.mask  # **Fix lỗi: Lưu output để lớp sau có thể sử dụng**
        return self.output
    
    def backward(self, d_loss):
        print(f"[DEBUG] Dropout backward input shape: {d_loss.shape}")  # 🛠️ Debug

        self.dinputs = d_loss * self.mask  # 🔥 Fix lỗi: Lưu gradient lại
        print(f"[DEBUG] Dropout dinputs shape: {self.dinputs.shape}")  # 🛠️ Debug

        return self.dinputs



    def eval(self):
        """Chuyển sang chế độ inference (không Dropout)."""
        self.training = False

    def train(self):
        """Chuyển sang chế độ training (có Dropout)."""
        self.training = True


class DataLoader:
    def __init__(self, images, labels, batch_size=32, shuffle=True, drop_last=False):
        """
        DataLoader để tải dữ liệu theo batch.
        
        :param images: NumPy array hoặc list chứa ảnh.
        :param labels: NumPy array hoặc list chứa nhãn.
        :param batch_size: Kích thước batch.
        :param shuffle: Có trộn dữ liệu hay không.
        :param drop_last: Nếu True, bỏ batch cuối nếu không đủ batch_size.
        """
        self.images = np.array(images, dtype=np.float32)  # Chuyển về float32
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.indexes = np.arange(len(self.images))

    def __len__(self):
        """Trả về số batch có thể lấy được."""
        total_batches = len(self.images) // self.batch_size
        if not self.drop_last and len(self.images) % self.batch_size != 0:
            total_batches += 1  # Thêm batch lẻ nếu không drop_last
        return total_batches

    def __iter__(self):
        """Khởi tạo iterator."""
        if self.shuffle:
            np.random.shuffle(self.indexes)
        self.current_index = 0
        return self

    def __next__(self):
        """Lấy batch tiếp theo."""
        if self.current_index >= len(self.images):
            raise StopIteration

        batch_indexes = self.indexes[self.current_index:self.current_index + self.batch_size]
        batch_images = self.images[batch_indexes]
        batch_labels = self.labels[batch_indexes]

        self.current_index += self.batch_size

        # Nếu drop_last và batch không đủ size, bỏ qua batch này
        if self.drop_last and batch_images.shape[0] < self.batch_size:
            raise StopIteration

        return batch_images, batch_labels

class BinaryCrossEntropy:
    def __init__(self, reduction='mean'):
        self.reduction = reduction
        self.trainable_layers = []  # Lưu các layer có trọng số
        self.accumulated_sum = 0
        self.accumulated_count = 0

    def remember_trainable_layers(self, trainable_layers):
        """Lưu danh sách các layer có trọng số để tính regularization."""
        self.trainable_layers = trainable_layers

    def new_pass(self):
        """Reset loss tích lũy trước mỗi epoch."""
        self.accumulated_sum = 0
        self.accumulated_count = 0

    def forward(self, y_pred, y_true):
        print(f"[DEBUG] y_pred shape: {y_pred.shape}")  # 🔍 Kiểm tra shape
        print(f"[DEBUG] y_true shape before reshape: {y_true.shape}")  # 🔍 Kiểm tra trước reshape
        print(f"[DEBUG] y_pred (min: {y_pred.min()}, max: {y_pred.max()})")
        print(f"[DEBUG] y_true unique values: {np.unique(y_true)}")  # Kiểm tra nhãn
        # Đảm bảo `y_true` có shape (batch_size, 1)
        if y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)

        print(f"[DEBUG] y_true shape after reshape: {y_true.shape}")  # 🔍 Kiểm tra sau reshape
        assert y_true.shape == y_pred.shape, "[ERROR] y_true shape không khớp với y_pred!"

        epsilon = 1e-9
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = - (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

        return np.mean(loss)

    def backward(self, y_pred, y_true):
        epsilon = 1e-9  # Tránh chia cho 0
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        self.dinputs = -(y_true / y_pred - (1 - y_true) / (1 - y_pred)) / len(y_true)  # 🔥 Fix: Lưu `self.dinputs`


    def calculate(self, output, y, include_regularization=False):
        """
        Tính loss và hỗ trợ regularization nếu có.
        """
        data_loss = self.forward(output, y)
        reg_loss = 0
        if include_regularization:
            for layer in self.trainable_layers:
                if hasattr(layer, 'weights'):
                    reg_loss += np.sum(layer.weights ** 2) * 0.0001  # L2 Regularization

        return data_loss + reg_loss

    def calculate_accumulated(self):
        """Trả về loss trung bình sau khi accumulate qua nhiều batch."""
        if self.accumulated_count == 0:
            return 0
        return self.accumulated_sum / self.accumulated_count

    
class SGD:
    def __init__(self, learning_rate=0.01, momentum=0.0, weight_decay=0.0, nesterov=False):
        """
        Tối ưu hóa SGD với Momentum, Weight Decay và Nesterov Momentum.
        
        :param learning_rate: Tốc độ học.
        :param momentum: Hệ số momentum (0 = không dùng momentum).
        :param weight_decay: Hệ số weight decay (L2 Regularization).
        :param nesterov: Sử dụng Nesterov momentum nếu True.
        """
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.velocity = {}  # Lưu vết momentum

    def pre_update_params(self):
        """Gọi trước khi cập nhật tham số (có thể thêm decay)."""
        pass

    def update_params(self, layer):
        """
        Cập nhật trọng số của layer theo SGD.
        
        :param layer: Layer cần cập nhật (phải có `weights` và `biases`).
        """
        if not hasattr(layer, 'weights') or not hasattr(layer, 'dweights'):
            return  # Bỏ qua nếu layer không có trọng số

        # Weight decay (L2 Regularization)
        if self.weight_decay:
            layer.dweights += self.weight_decay * layer.weights

        # Khởi tạo velocity nếu chưa có
        if layer not in self.velocity:
            self.velocity[layer] = {
                "w": np.zeros_like(layer.weights),
                "b": np.zeros_like(layer.biases)
            }

        # Lấy velocity hiện tại
        v_w, v_b = self.velocity[layer]["w"], self.velocity[layer]["b"]

        # Momentum update
        if self.momentum > 0:
            v_w = self.momentum * v_w - self.learning_rate * layer.dweights
            v_b = self.momentum * v_b - self.learning_rate * layer.dbiases

            if self.nesterov:
                layer.weights += self.momentum * v_w - self.learning_rate * layer.dweights
                layer.biases += self.momentum * v_b - self.learning_rate * layer.dbiases
            else:
                layer.weights += v_w
                layer.biases += v_b

            # Cập nhật velocity
            self.velocity[layer]["w"], self.velocity[layer]["b"] = v_w, v_b
        else:
            # SGD thường
            layer.weights -= self.learning_rate * layer.dweights
            layer.biases -= self.learning_rate * layer.dbiases

    def post_update_params(self):
        """Gọi sau khi cập nhật tham số (có thể thêm adaptive LR)."""
        pass

class Accuracy:
    def __init__(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0

    def new_pass(self):
        """Reset accuracy tích lũy trước mỗi epoch."""
        self.accumulated_sum = 0
        self.accumulated_count = 0

    def init(self, y_true):
        """Khởi tạo lại các giá trị tích lũy."""
        self.new_pass()

    def calculate(self, predictions, y_true):
        """Tính accuracy dựa trên dự đoán."""
        predictions = predictions > 0.5  # Chuyển thành dạng nhị phân
        accuracy = np.mean(predictions == y_true)

        # Cộng dồn accuracy để tính trung bình sau này
        self.accumulated_sum += accuracy
        self.accumulated_count += 1

        return accuracy

    def calculate_accumulated(self):
        """Trả về giá trị accuracy trung bình qua nhiều batch."""
        if self.accumulated_count == 0:
            return 0
        return self.accumulated_sum / self.accumulated_count


class Layer_Input:
    def __init__(self, input_shape=None):
        """
        Lớp đầu vào để nhận dữ liệu đầu vào vào mạng.
        
        :param input_shape: Tuple mô tả kích thước đầu vào (tùy chọn).
        """
        self.input_shape = input_shape

    def forward(self, inputs):
        """
        Forward pass - chỉ cần lưu lại đầu vào.
        
        :param inputs: Dữ liệu đầu vào của mạng.
        """
        self.output = inputs

    def backward(self, d_output):
        """
        Backward pass - không làm gì vì đây là lớp đầu vào.
        
        :param d_output: Gradient đầu ra từ lớp tiếp theo (không sử dụng ở đây).
        """
        self.dinputs = d_output  # Chỉ cần truyền gradient tiếp tục về phía trước

class Flatten:
    def forward(self, input_tensor, training=False):
        """
        Chuyển đổi tensor đầu vào thành dạng vector phẳng.
        :param input_tensor: NumPy array có shape (batch_size, C, H, W)
        :return: NumPy array có shape (batch_size, C * H * W)
        """
        self.input_shape = input_tensor.shape  # 🔥 Lưu shape để dùng trong backward
        self.output = input_tensor.reshape(input_tensor.shape[0], -1)  # Chuyển thành vector 2D
        return self.output

    def backward(self, d_loss):
        """
        Truyền ngược gradient qua Flatten.
        :param d_loss: Gradient từ lớp tiếp theo
        :return: Gradient reshaped về lại kích thước ban đầu
        """
        print(f"[DEBUG] Flatten backward input shape: {d_loss.shape}")  # 🛠️ Debug

        self.dinputs = d_loss.reshape(self.input_shape)  # 🔥 Khôi phục shape ban đầu
        print(f"[DEBUG] Flatten dinputs shape: {self.dinputs.shape}")  # 🛠️ Debug

        return self.dinputs



class Model:
    def __init__(self):
        self.layers = []
        self.trainable_layers = []
        self.input_layer = None
        self.output_layer_activation = None
        self.loss = None
        self.optimizer = None
        self.accuracy = None

    def add(self, layer):
        """Thêm một lớp vào mô hình."""
        self.layers.append(layer)
    
    def set(self, *, loss, optimizer, accuracy):
        """Thiết lập loss, optimizer và accuracy metric."""
        self.loss = loss
        self.optimizer = optimizer
        self.accuracy = accuracy

    def finalize(self):
        if not self.layers:
            raise ValueError("Model has no layers. Please add layers before finalizing.")
        if self.loss is None:
            raise ValueError("Loss function is not set. Use model.set(loss=...) before finalizing.")

        self.input_layer = Layer_Input()

        for i in range(len(self.layers)):
            if i == 0:
                self.layers[i].prev = self.input_layer
            else:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i - 1].next = self.layers[i]

        self.output_layer_activation = self.layers[-1]
        self.trainable_layers = [layer for layer in self.layers if hasattr(layer, 'weights')]

        self.loss.remember_trainable_layers(self.trainable_layers)

    def forward(self, X, training=False):
        """Truyền dữ liệu qua mạng (forward pass)."""
        print("[DEBUG] Forward start...")

        self.input_layer.forward(X)
        print(f"[DEBUG] Input shape: {X.shape}")

        for i, layer in enumerate(self.layers):
            print(f"[DEBUG] Forward over layer {i}: {layer.__class__.__name__}")

            try:
                layer.forward(layer.prev.output, training)
            except Exception as e:
                print(f"[ERROR] Issue at layer {i}: {layer.__class__.__name__} - {e}")
                raise e  # Dừng chương trình và hiển thị lỗi

        print("[DEBUG] Forward finished.")
        return self.output_layer_activation.output


    def backward(self, output, y_true):
        """Truyền ngược lỗi qua mạng (backward pass)."""

        print("[DEBUG] Backward start...")

        # Backpropagate qua Loss
        self.loss.backward(output, y_true)

        # Backpropagate qua lớp cuối cùng trước (Sigmoid)
        self.output_layer_activation.backward(self.loss.dinputs)

        # Truyền ngược qua các lớp còn lại
        for layer in reversed(self.layers[:-1]):  # Loại bỏ lớp cuối Sigmoid
            print(f"[DEBUG] Backward over layer {layer.__class__.__name__}")

            try:
                layer.backward(layer.next.dinputs)
            except Exception as e:
                print(f"[ERROR] issue at layer {layer.__class__.__name__}: {e}")
                raise e  # In lỗi chi tiết


    def train(self, X, y, *, epochs=1, batch_size=None, print_every=1, validation_data=None):
        self.accuracy.init(y)

        train_steps = len(X) // batch_size if batch_size else 1
        if batch_size and train_steps * batch_size < len(X):
            train_steps += 1

        print(f"Train steps per epoch: {train_steps}, Batch size: {batch_size}")

        for epoch in range(1, epochs + 1):
            print(f"\nEpoch {epoch}/{epochs}")
            self.loss.new_pass()
            self.accuracy.new_pass()

            for step in range(train_steps):
                start_time = time.time()

                print(f"Preparing batch {step+1}/{train_steps}...")
                batch_X = X[step * batch_size:(step + 1) * batch_size] if batch_size else X
                batch_y = y[step * batch_size:(step + 1) * batch_size] if batch_size else y

                print(f"Batch shape: {batch_X.shape}, Labels shape: {batch_y.shape}")
                
                if batch_y.ndim == 1:
                    batch_y = batch_y.reshape(-1, 1)
                
                print("Running forward pass...")
                output = self.forward(batch_X, training=True)
                print("Forward pass done.")

                print("Calculating loss and accuracy...")
                loss = self.loss.calculate(output, batch_y, include_regularization=True)
                accuracy = self.accuracy.calculate(self.output_layer_activation.predictions(output), batch_y)
                print(f"Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

                print("Running backward pass...")
                self.backward(output, batch_y)
                print("Backward pass done.")

                if self.optimizer:
                    print("Updating parameters...")
                    self.optimizer.pre_update_params()
                    for layer in self.trainable_layers:
                        self.optimizer.update_params(layer)
                    self.optimizer.post_update_params()
                    print("Parameters updated.")

                end_time = time.time()
                print(f"Step {step+1}/{train_steps} - Time: {end_time - start_time:.4f} sec")
                
                if step % print_every == 0 or step == train_steps - 1:
                    print(f"Step {step}/{train_steps}, acc: {accuracy:.3f}, loss: {loss:.3f}")

            print(f"Epoch {epoch} completed - acc: {self.accuracy.calculate_accumulated():.3f}, loss: {self.loss.calculate_accumulated():.3f}")

            if validation_data:
                print("Running validation...")
                self.evaluate(*validation_data, batch_size=batch_size)
                print("Validation completed.")


    def evaluate(self, X_val, y_val, *, batch_size=None):
        validation_steps = len(X_val) // batch_size if batch_size else 1
        if batch_size and validation_steps * batch_size < len(X_val):
            validation_steps += 1

        self.loss.new_pass()
        self.accuracy.new_pass()

        for step in range(validation_steps):
            batch_X = X_val[step * batch_size:(step + 1) * batch_size] if batch_size else X_val
            batch_y = y_val[step * batch_size:(step + 1) * batch_size] if batch_size else y_val

            output = self.forward(batch_X, training=False)
            self.loss.calculate(output, batch_y)
            self.accuracy.calculate(self.output_layer_activation.predictions(output), batch_y)

        print(f'Validation - Accuracy: {self.accuracy.calculate_accumulated():.3f}, Loss: {self.loss.calculate_accumulated():.3f}')

    def predict(self, X, *, batch_size=None):
        prediction_steps = len(X) // batch_size if batch_size else 1
        if batch_size and prediction_steps * batch_size < len(X):
            prediction_steps += 1

        output = []
        for step in range(prediction_steps):
            batch_X = X[step * batch_size:(step + 1) * batch_size] if batch_size else X
            batch_output = self.forward(batch_X, training=False)
            output.append(batch_output)

        return np.vstack(output)

    def save(self, path):
        """Lưu mô hình vào file."""
        model = copy.deepcopy(self)
        model.loss.new_pass()
        model.accuracy.new_pass()

        for layer in model.layers:
            for attr in ['inputs', 'output', 'dinputs', 'dweights', 'dbiases']:
                layer.__dict__.pop(attr, None)

        with open(path, 'wb') as f:
            pickle.dump(model, f)

    @staticmethod
    def load(path):
        """Tải mô hình từ file."""
        with open(path, 'rb') as f:
            return pickle.load(f)


class VGG16(Model):
    def __init__(self, num_classes=1, lr=0.01):
        super().__init__()

        self.add(Conv2D(3, 64, kernel_size=3, stride=1, padding=1, lr=lr))
        self.add(ReLU())
        self.add(Conv2D(64, 64, kernel_size=3, stride=1, padding=1, lr=lr))
        self.add(ReLU())
        self.add(MaxPool2D(kernel_size=2, stride=2))

        self.add(Conv2D(64, 128, kernel_size=3, stride=1, padding=1, lr=lr))
        self.add(ReLU())
        self.add(Conv2D(128, 128, kernel_size=3, stride=1, padding=1, lr=lr))
        self.add(ReLU())
        self.add(MaxPool2D(kernel_size=2, stride=2))

        self.add(Conv2D(128, 256, kernel_size=3, stride=1, padding=1, lr=lr))
        self.add(ReLU())
        self.add(Conv2D(256, 256, kernel_size=3, stride=1, padding=1, lr=lr))
        self.add(ReLU())
        self.add(Conv2D(256, 256, kernel_size=3, stride=1, padding=1, lr=lr))
        self.add(ReLU())
        self.add(MaxPool2D(kernel_size=2, stride=2))

        self.add(Conv2D(256, 512, kernel_size=3, stride=1, padding=1, lr=lr))
        self.add(ReLU())
        self.add(Conv2D(512, 512, kernel_size=3, stride=1, padding=1, lr=lr))
        self.add(ReLU())
        self.add(Conv2D(512, 512, kernel_size=3, stride=1, padding=1, lr=lr))
        self.add(ReLU())
        self.add(MaxPool2D(kernel_size=2, stride=2))

        self.add(Conv2D(512, 512, kernel_size=3, stride=1, padding=1, lr=lr))
        self.add(ReLU())
        self.add(Conv2D(512, 512, kernel_size=3, stride=1, padding=1, lr=lr))
        self.add(ReLU())
        self.add(Conv2D(512, 512, kernel_size=3, stride=1, padding=1, lr=lr))
        self.add(ReLU())
        self.add(MaxPool2D(kernel_size=2, stride=2))

        self.add(AdaptiveAvgPool2D((7, 7)))
        self.add(Flatten())

        self.add(Linear(512 * 7 * 7, 4096))
        self.add(BatchNorm1D(4096))  # Thêm BatchNorm
        self.add(ReLU())
        self.add(Dropout(0.5))

        self.add(Linear(4096, 4096))
        self.add(BatchNorm1D(4096))  # Thêm BatchNorm
        self.add(ReLU())
        self.add(Dropout(0.5))

        self.add(Linear(4096, num_classes))
        self.add(Sigmoid())

    def forward(self, x, training=False):
        return super().forward(x, training=training)

    def backward(self, output, y_true):
        return super().backward(output, y_true)


##############################################################################################


# Hàm load ảnh từ thư mục
def load_images_from_folder(folder, label, image_size=(224, 224)):
    images = []
    labels = []
    
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)  # Đọc ảnh
        if img is not None:
            img = cv2.resize(img, image_size)  # Resize về 224x224
            img = img / 255.0  # Chuẩn hóa về [0,1]
            img = np.transpose(img, (2, 0, 1))  # Chuyển từ (224, 224, 3) -> (3, 224, 224)
            images.append(img)
            labels.append(label)  # Giữ nguyên nhãn dạng int
    
    return images, labels

# Mở file để ghi
log_file = open("output13.log", "w")

# Ghi cả stdout và stderr vào file
sys.stdout = log_file
sys.stderr = log_file

print("Run program...")  # Câu này sẽ ghi vào file

# Check RAM status
available_memory = psutil.virtual_memory().available / (1024 * 1024)
print(f"[INFO] Available RAM: {available_memory:.2f} MB")

# Khởi tạo Model
model = VGG16(num_classes=1, lr=0.01)

# Set loss, optimizer, accuracy
loss_function = BinaryCrossEntropy()
optimizer = SGD(learning_rate=0.001)
accuracy = Accuracy()

# Cấu hình model
model.set(loss=loss_function, optimizer=optimizer, accuracy=accuracy)
model.finalize()

# Load dữ liệu từ thư mục
meningioma_images, meningioma_labels = load_images_from_folder('C:/Personal/final_graduate/data/meningioma', label=1)
non_meningioma_images, non_meningioma_labels = load_images_from_folder('C:/Personal/final_graduate/data/notumor', label=0)


# Gộp tất cả dữ liệu lại và shuffle
images = np.array(meningioma_images + non_meningioma_images)
labels = np.array(meningioma_labels + non_meningioma_labels)

print(f"Total images: {len(images)}, Total labels: {len(labels)}")
print(f"Image shape: {images[0].shape}")

# Shuffle dữ liệu
indices = np.arange(len(images))
np.random.shuffle(indices)
images, labels = images[indices], labels[indices]

# Train mô hình (Sử dụng hàm train có sẵn trong Model)
model.train(images, labels, epochs=10, batch_size=8, print_every=1)

# Lưu mô hình sau khi train
model.save("brain_tumor_vgg16.pkl")


log_file.close()  # Đóng file sau khi hoàn tất
