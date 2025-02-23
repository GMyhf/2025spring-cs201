import tiktoken

# 使用cl100k_base编码器，适用于GPT-3等新模型
encoding = tiktoken.get_encoding("cl100k_base")

# 中文文本
text = "北京大学"

# 对文本进行编码
encoded_tokens = encoding.encode(text)

print("Original text:", text)
print("Encoded tokens:", encoded_tokens)

# 解码回文本以验证
decoded_text = encoding.decode(encoded_tokens)
print("Decoded text from tokens:", decoded_text)

# 打印每个token及其对应的字符串表示
for token_id in encoded_tokens:
    print(f"Token ID: {token_id}, Token string: {repr(encoding.decode_single_token_bytes(token_id))}")

# 手动解码每个token的字节表示
token_bytes = [
    b'\xe5\x8c\x97\xe4\xba\xac', # 北京
    b'\xe5\xa4\xa7',             # 大
    b'\xe5\xad\xa6'              # 学
]

for byte_seq in token_bytes:
    print(byte_seq.decode('utf-8'))
