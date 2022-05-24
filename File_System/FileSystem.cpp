#pragma once

// Libraries
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstdlib>
#include <math.h> /* ceil */
#include <algorithm>

#include "FileSystem.SupportFunc.cpp" // *required*

class FileSystem
{
private:
    int BlockSize = 256;          // default Block size is 256 bits (characters)
    int DirectoryTableSize = 256; // in units of blocks
    int VolumeSize = 4096;        // in units of blocks

protected:
    // @description: sub-routine transfers specified table from storage to memeory
    // @param table: takes arugement 0 for FAT and 1 for DirectoryTable
    // @return an std::vector<std::vector<std::string>> with the specified table contents inside
    auto TabletoMemory(int table = -1) const
    {
        // error handling
        if (table == -1)
            throw std::invalid_argument("No argument supplied to TabletoMemory()");

        std::vector<std::vector<std::string>> Table;
        std::vector<std::string> buffer;

        std::fstream file;
        file.open("volume.txt", std::ios::in);
        std::string line;
        if (file.is_open())
        {
            getline(file, line);
            if (table == 0)
            {
                line = line.substr(0, this->DirectoryTableSize * this->BlockSize);
            }
        }

        std::string temp = "";
        for (auto &i : line)
        {
            if (i == ',')
            {
                buffer.push_back(temp);
                temp = "";
                continue;
            }
            else if (i == '|')
            {
                buffer.push_back(temp);
                temp = "";
                Table.push_back(buffer);
                buffer.clear();
                continue;
            }
            temp += i;
        }
        return Table;
    }

public:
    FileSystem(const char *FileName = "volume.txt")
    {
        srand(time(0));
        BuildVolume(FileName);
        BuildDirectoryTable();
    }

    // @description: Makes the storage volume text file and populates empty cells with 0
    // @param fname: volume name
    void BuildVolume(const char *fname = "volume.txt") const
    {
        std::cout << fname << " is being created..." << std::endl;
        // remove file if it exists
        if (remove(fname) == 0)
            printf("The file is deleted successfully...\n");
        else
            printf("The file is not deleted...\n");

        // build file
        unsigned int size = this->VolumeSize * this->BlockSize;

        std::string text("");
        for (int i = 0; i < size; i++)
        {
            // std::string s = "0123456789";
            // char c = s[rand() % 10];
            // text += c;
            text += '0';
        }
        text += '\n';

        // create and build files
        std::string volume(fname);
        std::fstream outfile;
        outfile.open(volume, std::ios_base::app);
        if (!outfile.is_open())
        {
            std::cerr << "failed to open... " << volume << '\n';
        }
        else
        {
            outfile.write(text.data(), text.size());
            std::cout << "Done Writing Volume!..." << std::endl;
        }
    }

    // @description: Builds, encodes and writes the Directory Table into the head of the volume
    void BuildDirectoryTable() const
    {
        std::string text = "00000000000.DirectoryTable,0000|";

        for (int i = 1; i < this->DirectoryTableSize * this->BlockSize / 32; i++)
            text += "00000000000000000000000000,0000|";

        std::fstream file;
        file.open("volume.txt");
        file.seekp(0, std::ios::beg);
        file.write(text.c_str(), text.size());
        file.close();
    }

    // @description: Finds Free Blocks in the volume
    // @return: the index of the block in decimal
    int FindFreeBlock() const
    {
        std::fstream file;
        file.open("volume.txt");
        std::string line;
        if (file.is_open())
        {
            getline(file, line);
            line = line.substr(this->DirectoryTableSize * this->BlockSize);
        }
        file.close();

        std::vector<int> FreeBlocks;
        for (int i = 0; i < line.size(); i += this->BlockSize)
        {
            if (line[i] == '0')
                FreeBlocks.push_back((i / this->BlockSize) + this->BlockSize);
        }

        int Range = FreeBlocks.size() - 1 - this->BlockSize + 1;
        int Random = rand() % Range + 1;

        return FreeBlocks[Random];
    }

    // @descriptions add entry into the directory table and and writes input dynamically into the volume
    // @param: FileName is the name of the file to be added into the directory table
    // @param: input is the input of the string to be written into the volume
    void WriteWrapper(std::string FileName, std::string input) const
    {
        // base case if filename exceeds length limit
        if (FileName.length() > 26)
        {
            std::cerr << "Max FileName length is 26" << std::endl;
            return;
        }

        // load DT into memory
        auto DT = GetDirectoryTable();

        // check if the filename already exists
        for (auto i : DT)
        {
            std::string str = i[0];
            if (str.find(FileName) != std::string::npos)
            {
                std::cerr << "err: duplicate filename" << std::endl;
                return;
            }
        }

        // insert filename into the DT
        int DTEntryIndex;
        for (int i = 0; i < DT.size(); i++)
        {
            if (DT[i][0] == "00000000000000000000000000")
            {
                DTEntryIndex = i;
                std::string fname;
                for (int i = 0; i < (26 - FileName.length()); i++)
                    fname += '0';

                DT[i][0] = fname + FileName;
                break;
            }
        }

        // insert index block into DT
        std::string hex = DecToHex(FindFreeBlock());
        DT[DTEntryIndex][1] = hex;
        OverwriteDirectoryTable(DT);

        // insert all other indecies into the the Index Block in the writable volume
        std::fstream file;
        file.open("volume.txt");
        std::string line;
        if (file.is_open())
        {
            getline(file, line);
            line = line.substr(this->DirectoryTableSize * this->BlockSize);
        }
        file.close();

        int BlocksNeeded = ceil(float(input.length()) / float(this->BlockSize));
        std::string IndexBuffer = "";
        for (int i = 0; i < BlocksNeeded; i++)
        {
            std::string temp = DecToHex(FindFreeBlock());
            IndexBuffer += temp;
        }

        // debug print
        //  std::cout << IndexBuffer << std::endl;

        if (IndexBuffer.length() < this->BlockSize)
        {
            for (int i = 0; i < IndexBuffer.length(); i++)
                line[HexToDec(hex) * this->BlockSize + i] = IndexBuffer[i];
        }
        else
        {
            std::vector<std::string> IndexBlocks;
            std::string tmp = "";
            for (int i = 0; i < IndexBuffer.length(); i += this->BlockSize - 4)
            {
                std::string temp = IndexBuffer.substr(i, this->BlockSize - 4);
                tmp += (temp);
                tmp += (DecToHex(FindFreeBlock()));
                IndexBlocks.push_back(tmp);
                tmp = "";
            }
            std::string LastSubString = IndexBlocks[IndexBlocks.size() - 1];
            LastSubString = LastSubString.substr(0, LastSubString.length() - 4);
            IndexBlocks[IndexBlocks.size() - 1] = LastSubString;

            // write to DT

            // write to volume
            for (int i = 0; i < IndexBlocks.size(); i++)
            {
                std::string r = IndexBlocks[i].substr(0, this->BlockSize - 4);
                std::string s = IndexBlocks[i].substr(IndexBlocks[i].length() - 4 - 1, 4);
                // WriteToVolume(HexToDec(s), r+s);
            }

            // return;

            // Debugging prints
            // for (auto i : IndexBlocks)
            //     for (auto j : i)
            //         std::cout << j;

            // std::cout << std::endl;
        }

        file.open("volume.txt");
        file.seekp(this->BlockSize * this->DirectoryTableSize, std::ios::beg);
        file.write(line.c_str(), line.size());
        file.close();

        // insert the data stream into the blocks
        for (int i = 0; i < IndexBuffer.length(); i += 4)
        {
            std::string s = IndexBuffer.substr(i, 4);
            WriteToVolume(HexToDec(s), input.substr((i / 4) * this->BlockSize, this->BlockSize));
        }
    }

    // @descriptions add entry into the directory table and and writes input dynamically into the volume
    // @param: FileName is the name of the file to be added into the directory table
    // @param: input is the input of the string to be written into the volume
    void WriteWrapperNew(std::string FileName, std::string input) const
    {
        std::vector<int> FreeBlocks;
        int FreeBlocksNeeded = ceil(float(input.length()) / float(this->BlockSize));

        for (int i = 0; i < FreeBlocksNeeded + 1; i++)
        {
            int Free = FindFreeBlock();
            if (std::count(FreeBlocks.begin(), FreeBlocks.end(), Free))
                i--;
            else
                FreeBlocks.push_back(Free);
        }
        std::string FreeBlocksStr = "";
        for (int i = 1; i < FreeBlocks.size(); i++)
            FreeBlocksStr += DecToHex(FreeBlocks[i]);

        // write index block to DT and Volume
        std::string buffer = "";
        for (int i = 0; i < 26 - FileName.length(); i++)
            buffer += "0";
        WriteToDirectoryTable(buffer + FileName, DecToHex(FreeBlocks[0]));
        WriteToVolume(FreeBlocks[0], FreeBlocksStr);

        // write to volume
        for (int i = 0; i < input.length() / this->BlockSize; i++)
        {
            std::string substr = input.substr(i, this->BlockSize);
            WriteToVolume(FreeBlocks[i + 1], substr);
        }
    }

    // @descriptions: deletes entries in the Directory Table (soft delete)
    // @param FileName: the specifier for determining the file to be delete
    void Delete(std::string FileName) const
    {
        auto DT = GetDirectoryTable();

        for (int i = 0; i < DT.size(); i++)
        {
            if (DT[i][0].find(FileName) != std::string::npos)
            {
                DT[i][0] = "00000000000000000000000000";
                DT[i][1] = "0000";
            }
        }
        OverwriteDirectoryTable(DT);
    }

    // @description: Read operation for file system
    // @param FileName: the specifier for determining the file to be read
    // @return a string of the data in the file
    std::string Read(std::string FileName) const
    {
        auto DT = GetDirectoryTable();
        int block = 0;
        for (int i = 0; i < DT.size(); i++)
            if (DT[i][0].find(FileName) != std::string::npos)
                block = HexToDec(DT[i][1]);

        std::string Blocks = GetBlock(block + this->BlockSize);
        std::string line = "";
        for (int i = 0; i < Blocks.length(); i += 4)
        {
            std::string s = Blocks.substr(i, 4);
            if (s == "0000")
                break;
            line += GetBlock(HexToDec(s));
        }

        return line;
    }

    // @description: Read operation for file system
    // @param FileName: the specifier for determining the file to be read
    // @return a string of the data in the file
    std::string ReadBig(std::string FileName) const
    {
        auto DT = GetDirectoryTable();
        int block = 0;
        for (int i = 0; i < DT.size(); i++)
            if (DT[i][0].find(FileName) != std::string::npos)
                block = HexToDec(DT[i][1]);

        // std::string Blocks = GetBlock(block + this->BlockSize); // keeping this here as a toggle
        std::string Blocks = GetBlock(block);
        std::string line = "";

        if (Blocks.substr(Blocks.length() - 4, 4) == "0000")
        {
            for (int i = 0; i < Blocks.length(); i += 4)
            {
                std::string s = Blocks.substr(i, 4);
                if (s == "0000")
                    break;
                line += GetBlock(HexToDec(s));
            }
        }
        else
        {
            std::vector<std::string> indecies;
        }

        return line;
    }

    // @description: FileSystems commands...
    void FileSystemCommand() const
    {
        std::string s = "";
        while (true)
        {
            std::cout << "\033[1;31m╭─user@FileSystem\033[0m\n\033[1;31m╰─$ \033[0m";
            std::cin >> s;

            std::vector<std::string> input = extract_strings(s);

            if (input[0] == "exit")
                break;
            else
                std::cout << "command not found: " << s << std::endl;
        }
    }

    // @description overwrites DirectoryTable with param
    // @param a vector of vector strings that will be used to overwrite the DirectoryTable
    void OverwriteDirectoryTable(const std::vector<std::vector<std::string>> &table) const
    {
        std::string text = "";

        for (int i = 0; i < table.size(); i++)
        {
            for (int j = 0; j < table[i].size(); j++)
            {
                text += table[i][j];
                if (j == 0)
                    text += ",";
            }
            text += "|";
        }

        int len = (GetDirectoryTableSize() * GetBlockSize()) - text.length();
        for (int i = 0; i < len; i++)
            text += '0';

        std::fstream file;
        file.open("volume.txt");
        file.seekp(0, std::ios::beg);
        file.write(text.c_str(), text.size());
        file.close();
    }

    void Write(std::string FileName) const
    {
        std::fstream file;
        file.open(FileName);
        std::string line;
        if (file.is_open())
        {
            getline(file, line);
        }
        WriteWrapperNew(FileName, line);
        file.close();
    }

    // @description: overwrites a block within the writable volume
    // @param BlockIndex: the index block which will be overwritten
    // @param Input: the input data stream in form string that will be used to overwrite MAXSIZE = 256
    void WriteToVolume(int BlockIndex, std::string Input) const
    {
        std::fstream file;
        file.open("volume.txt");
        file.seekp(BlockIndex * this->BlockSize, std::ios::beg);
        file.write(Input.c_str(), Input.size());
        file.close();
    }

    // @description: wirtes to an entry into the Directory Table given an address
    // @param FileName: name of the file for the entry
    // @param Address: address of the index location for the index block in the allocation (in hexadecimal)
    void WriteToDirectoryTable(std::string FileName, std::string Address) const
    {
        // base case if filename exceeds length limit
        if (FileName.length() > 26)
        {
            std::cerr << "Max FileName length is 26" << std::endl;
            return;
        }

        // store into memory
        auto DT = GetDirectoryTable();

        // check if the filename already exists
        for (auto i : DT)
        {
            std::string str = i[0];
            if (str.find(FileName) != std::string::npos)
            {
                std::cerr << "err: duplicate filename" << std::endl;
                return;
            }
        }

        // linearly find the next available table slot
        for (int i = 0; i < DT.size(); i++)
        {
            if (DT[i][0] == "00000000000000000000000000" and DT[i][1] == "0000")
            {
                std::string buffer = "";
                for (int j = 0; j < 26 - FileName.length(); j++)
                    buffer += '0';

                DT[i][0] = buffer + FileName;
                DT[i][1] = Address;
                break;
            }
        }

        OverwriteDirectoryTable(DT);
    }

    // @description: retrive the datastream of block given an index
    // @param BlockIndex: the block index wanted for reterival
    // @return the string stream of the data
    std::string GetBlock(int BlockIndex) const
    {
        std::fstream file;
        file.open("volume.txt");
        std::string line;
        if (file.is_open())
        {
            getline(file, line);
            int size = BlockIndex * this->BlockSize;
            line = line.substr(size, this->BlockSize);
        }
        file.close();

        return line;
    }

    // @description: wrapper method: transfers DirectoryTable from storage to memeory
    // @return a std::vector<std::vector<std::string>> with the specified table contents inside
    std::vector<std::vector<std::string>> GetDirectoryTable() const { return TabletoMemory(0); }

    // getters
    int GetBlockSize() const { return this->BlockSize; }
    int GetDirectoryTableSize() const { return this->DirectoryTableSize; }

    // setters
    void SetBlockSize(int size) { this->BlockSize = size; }
    void SetDirectoryTableSize(int size) { this->DirectoryTableSize = size; }
};
