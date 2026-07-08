#include <stdio.h>
#include <iostream>
#include <fstream>
#include "nlohmann/json.hpp"
#include <string>
#include <cmath>

using namespace std;
using json = nlohmann::json;

vector<vector<double>> matrix_mult(vector<vector<double>> A, vector<vector<double>> B) {
    int N = A.size();
    int D = B[0].size();
    int inner = A[0].size();

    vector<vector<double>> covish(N, vector<double>(D));

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < D; j++)
        {
            double sum = 0.0;

            for (int k = 0; k < inner; k++)
            {
                sum += A[i][k] * B[k][j];
            }

            covish[i][j] = sum;
        }
    }
    return covish;
}

vector<vector<double>> normalize(vector<vector<double>> v) {
    double norm_tot = 0;
    for (int i=0; i<v.size(); i++) {
        norm_tot += v[i][0]*v[i][0];
    }
    double val = sqrt(norm_tot);

    vector<vector<double>> v_copy(v.size(), {1});

    for (int i = 0; i < v.size(); i++) {
        v_copy[i][0] = v[i][0] / val;
    }

    return v_copy;
}

vector<vector<double>> power_iter(vector<vector<double>> A, int iters) {
    int N= A.size();
    vector<vector<double>> v(N, {1.0});

    v = normalize(v);

    for (int i=0; i<iters; i++) {
        v = matrix_mult(A, v);
        v = normalize(v);
    }

    return v;
}


int main() {
    
    ifstream file("pca_input.json");

    if (!file.is_open()) {
        cerr << "ERRROR OPENING FILE!" << endl;
        return 1;
    }

    json data = json::parse(file)

    vector<vector<double>> son_arrs_raw = data["vectors"];


    int N = son_arrs_raw.size();
    int D = son_arrs_raw[0].size();

    vector<double> means = vector<double>(D);

    for (int j=0; j<D;j++) {
        double sum=0;
        for (int i=0; i<N; i++) {
            sum+=son_arrs_raw[i][j];
        }
        means[j]=sum/(double)N;
    }

    vector<vector<double>> X_hat(N, vector<double>(D));

    for (int i=0; i<N; i++) {
        for (int j=0; j<D; j++) {
            X_hat[i][j] = son_arrs_raw[i][j]-means[j];
        }
    }

    // matrix multiplication (aka repeated dot products) to get 1000 x 1000 d matrix

    vector<vector<double>> covish(N, vector<double>(N));

    for (int i=0; i<N; i++) {
        for (int j = i; j < N; j++){
            double sum = 0.0;

            for (int k=0; k<D; k++) {
                sum += X_hat[i][k] * X_hat[j][k];
            }

            covish[i][j] = sum;
            covish[j][i] = sum;
        }
    }
    
    vector<vector<double>>e_1 = power_iter(covish, 100);

    double eigenvalue = 0.0;

    // compute A*v
    vector<double> Av(N, 0.0);

    for (int i = 0; i < N; i++){
        for (int j = 0; j < N; j++){
            Av[i] += covish[i][j] * e_1[j][0];
        }
    }

    // dot product vT * (A*v)
    for (int i = 0; i < N; i++)
    {
        eigenvalue += e_1[i][0] * Av[i];
    }

    cout << "eigenvalue: " << eigenvalue << endl;

    vector<vector<double>> X_hat_transpose(D, vector<double>(N));

    for (int i=0; i<N; i++) {
        for (int j=0; j<D; j++) {
            X_hat_transpose[j][i]=X_hat[i][j];
        }
    }

    vector<vector<double>>e_2 = matrix_mult(X_hat_transpose, e_1);

    vector<double> eigenvector(D, 0);

    for (int i=0; i<e_2.size(); i++) {
        eigenvector[i]=e_2[i][0];
        eigenvector[i] /= sqrt(eigenvalue);
    }

    cout << eigenvector[4] << endl;


    double important = 0;

    for (int i=0; i<eigenvector.size(); i++) {
        important += (eigenvector[i] * eigenvector[i]);
    }

    cout << "norm should be 1: " << important << endl;

    ofstream out("pc1.txt");

    if (!out.is_open())
    {
        cerr << "ERORROR" << endl;
        return 1;
    }

    for (double x : eigenvector)
    {
        out << x << "\n";
    }

    out.close();

    return 0;
}
