// Libraries
#include "FileSystem.cpp"

int main()
{
    FileSystem fs("volume.txt");
    std::cout << "Select a test number (1, 2 or 3): ";
    int TestSelector;
    std::cin >> TestSelector;

    // test 1: Machine Learning Data
    if (TestSelector == 1)
    {
        fs.Write("wdbc.data");
        std::cout << fs.ReadBig("wdbc.data") << std::endl;
    }
    // test 2: Small file (one index block)
    else if (TestSelector == 2)
    {
        fs.WriteWrapper("test", "Hello from test 3!");
        std::cout << fs.Read("test") << std::endl;
    }
    // test 3: Big file (multi index blocks)
    else if (TestSelector == 3)
    {
        fs.Write("test2.txt");
        std::cout << fs.ReadBig("test2.txt") << std::endl;
    }
}
