// sudo apt install openmpi-bin openmpi-common libopenmpi-dev
// mpic++ phonebook_search.cpp -o p
// mpirun -np 2 ./p input.txt Emma

// easy one for MPI string communication and searching in phonebook files

#include <bits/stdc++.h>
#include <mpi.h>
using namespace std;


void send_string(const string &text, int receiver) {
    int len = text.size() + 1;
    MPI_Send(&len, 1, MPI_INT, receiver, 1, MPI_COMM_WORLD);
    MPI_Send(text.c_str(), len, MPI_CHAR, receiver, 1, MPI_COMM_WORLD);
}

string receive_string(int sender) {
    int len;
    MPI_Recv(&len, 1, MPI_INT, sender, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    char *buf = new char[len];
    MPI_Recv(buf, len, MPI_CHAR, sender, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    // Construct a C++ string from the buffer
    string res(buf);
    delete[] buf;
    return res;
}

string vector_to_string(const vector<string> &lines, int start, int end) {
    string result;
    for (int i = start; i < min((int)lines.size(), end); i++) {
        result +=lines[i]+ "\n";
    }
    return result;
}

vector<string> string_to_lines (const string &text) {
    vector<string> lines;  // result vector 
    istringstream iss(text);    // stream to read text line by line
    string line;
    while (getline(iss, line)) {// read each line until end of text
        if(!line.empty()){
            lines.push_back(line);
        }    
    }
    return lines;
}

string check(const string &line, const string &search) {
    if (line.find(search)!= string::npos) {
        return line+"\n";
    }
    return "";
}

void read_phonebook(const vector<string> &files, vector<string>&lines){
    for (const string &file : files) {
        ifstream f(file);   // open file
        string line;
        if(!f.is_open()){
            cerr<<"Could not open file: "<< file <<endl;
            continue;
        }
        while (getline(f, line)) {  // read line by line
            if (!line.empty()){
                lines.push_back(line);
            }
        }
    }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 3) {
        if (rank == 0)
            cerr << "Usage: mpirun -n <procs> " << argv[0] << " <file>... <search>\n";
        MPI_Finalize();
        return 1;
    }

    string search_term = argv[argc - 1];
    double start, end;

    if (rank == 0) {
        vector<string> files(argv + 1, argv + argc - 1);
        vector<string> lines;
        read_phonebook(files, lines );
        int total = lines .size();
        int chunk = (total + size - 1) / size;

        for (int i = 1; i < size; i++) {
            string text = vector_to_string(lines , i * chunk, (i + 1) * chunk);
            send_string(text, i);
        }

        start = MPI_Wtime();
        string result;
        for (int i = 0; i < min(chunk, total); i++) {
            string match = check(lines[i], search_term);
            if (!match.empty()) result += match;
        }
        end = MPI_Wtime();

        for (int i = 1; i < size; i++) {
            string recv = receive_string(i);
            if (!recv.empty()) result += recv;
        }
        
        ofstream out("output.txt");
        out << result;
        out.close();
        printf("Total results found: %d\n", (int)count(result.begin(), result.end(), '\n'));
        printf("Process %d took %f seconds.\n", rank, end - start);

    }
    else {
        string recv_text = receive_string(0);
        vector<string>lines=string_to_lines (recv_text);
        start = MPI_Wtime();
        string result;
        for (auto &c : lines ) {
            string match = check(c, search_term);
            if (!match.empty()) result += match;
        }
        end = MPI_Wtime();
        send_string(result, 0);
        printf("Process %d took %f seconds.\n", rank, end - start);
    }

    MPI_Finalize();
    return 0;
}