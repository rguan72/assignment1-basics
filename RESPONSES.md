2.1
a) The null character, '\x00'.
b) It's string representation is "'\\x00'" but when you print it you just see blank. 
c) In text, it renders invisibly when printed to stdout even when the character exists in the string. 

2.2
a) For common strings (in English, which is most of the internet), UTF-8 encodings generally require fewer bits of data than UTF-16. This is because most common characters can be represented with one byte of data, so occassionally paying many bytes to represent rare characters is a worthwhile cost compared to representing every character with 2 or 4 bytes.
b) This function assumes every character can be represented with exactly one byte in UTF-8. It fails for: 
decode_utf_8_bytes_to_str_wrong("😀".encode("utf-8"))
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "<stdin>", line 1, in decode_utf_8_bytes_to_str_wrong
UnicodeDecodeError: 'utf-8' codec can't decode byte 0xf0 in position 0: unexpected end of data
c) bytes([255, 255]) does not decode because no utf-8 unicode character starts with 255. utf-8 has a scheme for showing how many bytes the character requires, and 11111111 is not a valid prefix since there are no 8 byte characters. We can also have a valid start byte but invalid continuation byte.