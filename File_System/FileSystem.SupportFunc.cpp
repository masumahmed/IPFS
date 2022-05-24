#pragma once

#include <iostream>
#include <string>
#include <fstream>

void print(std::string s)
{
    std::cout << s << std::endl;
}

// extract subsstrings of s using space delimiters
std::vector<std::string> extract_strings(std::string s)
{
    std::vector<std::string> result;
    std::string temp;

    for (auto &i : s)
    {
        if (i == ' ' and !temp.empty())
        {
            result.push_back(temp);
            temp = "";
        }
        else
            temp += i;
    }
    return result;
}

std::string DecimaltoBinary(int num)
{
    std::string buffer;
    while (num != 0)
    {
        buffer = (num % 2 == 0 ? '0' : '1') + buffer;
        num /= 2;
    }
    return buffer;
}

int HexToDec(std::string hex)
{
    int len = hex.size();
    int base = 1;
    int dec_val = 0;

    for (int i = len - 1; i >= 0; i--)
    {
        if (hex[i] >= '0' && hex[i] <= '9')
        {
            dec_val += (int(hex[i]) - 48) * base;
            base = base * 16;
        }
        else if (hex[i] >= 'A' && hex[i] <= 'F')
        {
            dec_val += (int(hex[i]) - 55) * base;
            base = base * 16;
        }
    }
    return dec_val;
}

std::string DecToHex(int num)
{
    char hex[100];

    int i = 0;
    while (num != 0)
    {
        int temp = 0;
        temp = num % 16;
        if (temp < 10)
        {
            hex[i] = temp + 48;
            i++;
        }
        else
        {
            hex[i] = temp + 55;
            i++;
        }

        num = num / 16;
    }

    std::string result = "";
    for (int j = i - 1; j >= 0; j--)
        result += hex[j];

    std::string ResultWithLeadingZeros = "";

    for (int i = 0; i < (4 - result.length()); i++)
        ResultWithLeadingZeros += "0";

    ResultWithLeadingZeros += result;

    return ResultWithLeadingZeros;
}
