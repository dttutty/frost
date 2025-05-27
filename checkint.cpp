#include <iostream>
#include <climits> // 包含 CHAR_BIT

int main() {
    // 1. 获取 int 占用的字节数
    size_t int_size_bytes = sizeof(int);
    std::cout << "Size of int: " << int_size_bytes << " bytes" << std::endl;

    // 2. 获取 int 占用的位数 (bits)
    // CHAR_BIT 定义了一个字节包含的位数 (通常是 8)
    size_t int_size_bits = sizeof(int) * CHAR_BIT;
    std::cout << "Size of int: " << int_size_bits << " bits" << std::endl;

    // 你也可以检查其他类型
    std::cout << "Size of short: " << sizeof(short) * CHAR_BIT << " bits" << std::endl;
    std::cout << "Size of long: " << sizeof(long) * CHAR_BIT << " bits" << std::endl;
    std::cout << "Size of long long: " << sizeof(long long) * CHAR_BIT << " bits" << std::endl;
    std::cout << "Size of float: " << sizeof(float) * CHAR_BIT << " bits" << std::endl;
    std::cout << "Size of double: " << sizeof(double) * CHAR_BIT << " bits" << std::endl;
    std::cout << "Size of pointer: " << sizeof(void*) * CHAR_BIT << " bits" << std::endl;

    return 0;
}
