# XLA_IT5_Group10

* Do Github giới hạn dung lượng project upload
* File dataset nhóm em dùng là EMNIST (Extended MNIST)
* File thuộc quyền sở hữu của Gregory Cohen, Saeed Afshar, Jonathan Tapson, and Andre van Schaik, The MARCS Institute for Brain, Behaviour and Development, Western Sydney University
* Tập dữ liệu EMNIST là tập hợp các chữ số ký tự viết tay có nguồn gốc từ NIST Special Database 19 của NIST và được chuyển đổi sang định dạng hình ảnh 28x28 pixel và cấu trúc tập dữ liệu khớp trực tiếp với tập dữ liệu MNIST
* Có sáu phần tách khác nhau được cung cấp trong tập dữ liệu này và mỗi phần được cung cấp ở hai định dạng:
      Binary (xem emnist_source_files.zip)
      CSV (nhãn và hình ảnh kết hợp)
            Mỗi hàng là một hình ảnh riêng biệt
            785 cột
            Cột đầu tiên = class_label (xem maps.txt để biết định nghĩa nhãn lớp)
            Mỗi cột sau đại diện cho một giá trị pixel (tổng cộng 784 cho hình ảnh 28 x 28)

* Phần bài sử dụng là EMNIST Balanced:  131,600 ký tự. 47 lớp balanced. Nó được lấy từ tập dữ liệu ByMerge để giảm các lỗi phân loại sai do chữ in hoa và chữ thường và cũng có số lượng mẫu bằng nhau trên mỗi lớp.
* Trước khi chạy predict.py, hãy download dataset https://drive.google.com/drive/folders/1z4Ls3boRTdK6Ykh5X5QbrVVOLSCb21MA?usp=sharing và unzip tại ../data


* Nhóm 10 gồm 5 thành viên: ![image](https://github.com/user-attachments/assets/51f7d776-bb02-4a73-96d0-7db00de3b9c3)
