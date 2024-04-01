#include <iostream>
#include "Eigen/Eigen"
#include <Eigen/QR>
#include <vector>
#include <cmath>

//Inizialmente, per quanto riguarda la fattorizzazione PALU, avevo cercato di scrivere una funzione (anzi, più di una)
//scrivendo le matrici sia con Matrix che con vector<vector, ma il programma mi restituiva errori relativi troppo grandi
//e non sono riuscita a capire come migliorare o correggere quello che avevo scritto (inoltre avevo scritto un sacco di
//comandi complicati); così ho trovato un modo alternativo per ottenere questo tipo di decomposizione, senza implementare
//una funzione e mi sembra che funzioni bene. Per quanto riguarda la fattorizzazione QR sono riuscita a scrivere una funzione.
//Per completezza ho lasciato il codice iniziale commentato sotto quello definitivo

using namespace std;
using namespace Eigen;

// costruisco la funzione che calcola l'erorre relativo
double rel_err(const VectorXd& w, const VectorXd& z ) {
    double norm_w = w.norm();
    double norm_z = z.norm();

    double norm_diff = (w - z).norm();

    // calcolo l'errore relativo
    double rell_err = norm_diff/ norm_z;

    return rell_err;
}
// scrivo la funzione per effettuare la decomposizione QR
void QR_decomposition(const MatrixXd& A, MatrixXd& Q, MatrixXd& R) {
    int n = A.rows();

    // calcolo la fattorizzazione QR utilizzando la classe HouseholderQR
    HouseholderQR<MatrixXd> qr(A);

    Q = qr.householderQ();
    R = qr.matrixQR().triangularView<Upper>();

    // normalizzo la matrice Q (le colonne di Q sono già ortonormali)
    for (int i = 0; i < n; ++i) {
        double norm_col = Q.col(i).norm();
        Q.col(i) /= norm_col;
        R.row(i) *= norm_col;
    }
}


int main()
{
    // definisco la soluzione comune ai tre sistemi
    Vector2d x = Vector2d:: Ones();
    x << -1.0e+0, -1.0e+00;
    //cout << "the solution of the three systems is:" << endl;
    //cout << x << endl;

    // descrivo A e b dei tre sistemi lineari Ax=b
    // di seguito li risolvo utilizzando la fattorizzazione PALU
    cout << "Fattorizzazione PALU";
    cout << "\n" << endl;

    // sistema 1

    MatrixXd A1(2, 2);
    A1 << 5.547001962252291e-01, -3.770900990025203e-02,
        8.320502943378437e-01, -9.992887623566787e-01;
    VectorXd b1(2);
    b1 << -5.169911863249772e-01, 1.672384680188350e-01;

    //cout << "A1 matrix:" << endl<< A1 << endl;
    //cout << "b1 (column vector):" << endl;
    //cout << b1 << endl;

    // fattorizzazione PALU della matrice A1
    PartialPivLU<MatrixXd> lu1(A1);

    // soluzione del sistema lineare A1x1 = b1
    VectorXd x1 = lu1.solve(b1);


    // stampo la soluzione
    cout << "The solution x1 is:\n" << x1 << endl;

    double PALU_rel_err1 = rel_err(x1, x);
    cout << "The relative error is: \n" << PALU_rel_err1 << endl;

    cout << "\n" << endl;

    // sistema 2

    MatrixXd A2(2, 2);
    A2 << 5.547001962252291e-01, -5.540607316466765e-01,
        8.320502943378437e-01, -8.324762492991313e-01;
    VectorXd b2(2);
    b2 << -6.394645785530173e-04, 4.259549612877223e-04;

    //cout << "A2 matrix:" << endl<< A2 << endl;
    //cout << "b2 (column vector):" << endl;
    //cout << b2 << endl;

    PartialPivLU<MatrixXd> lu2(A2);

    VectorXd x2 = lu2.solve(b2);

    cout << "The solution x2 is:\n" << x2 << endl;

    double PALU_rel_err2 = rel_err(x2, x);
    cout << "The relative error is: \n" << PALU_rel_err2 << endl;

    cout << "\n" << endl;

    // sistema 3

    MatrixXd A3(2, 2);
    A3 << 5.547001962252291e-01,-5.547001955851905e-01,
        8.320502943378437e-01, -8.320502947645361e-01;
    VectorXd b3(2);
    b3 << -6.400391328043042e-10, 4.266924591433963e-10;

    //cout << "A3 matrix:" << endl<< A3 << endl;
    //cout << "b3 (column vector):" << endl;
    //cout << b3 << endl;

    PartialPivLU<MatrixXd> lu3(A3);

    VectorXd x3 = lu3.solve(b3);

    cout << "The solution x3 is:\n" << x3 << endl;

    double PALU_rel_err3 = rel_err(x3, x);
    cout << "The relative error is: \n" << PALU_rel_err3 << endl;

    cout << "\n" << endl;

    // applico la fattorizzazione QR ai tre sistemi
    cout << "Fattorizzazione QR" << endl << "\n" << endl;

    MatrixXd Q1, R1;

    // calcolo la fattorizzazione QR
    QR_decomposition(A1, Q1, R1);

    // stampo le matrici Q e R
    //cout << "Matrice Q1:" << endl << Q1 << endl;
    //cout << "Matrice R1:" << endl << R1 << endl;

    // calcolo il vettore y = Q^Tb
    VectorXd y1_QR = Q1.transpose() * b1;

    // risolvo il sistema Rx = y per ottenere il vettore soluzione x
    VectorXd x1_QR = R1.inverse()*y1_QR;
    cout << "The solution x1 is:" << endl << x1_QR << endl;

    double QR_rel_err1 = rel_err(x1_QR,x);
    cout << "The relative error is: \n" << QR_rel_err1 << endl;

    cout << "\n";

    MatrixXd Q2, R2;

    QR_decomposition(A2, Q2, R2);

    //cout << "Matrice Q2:" << endl << Q2 << endl;
    //cout << "Matrice R2:" << endl << R2 << endl;

    VectorXd y2_QR = Q2.transpose() * b2;

    VectorXd x2_QR = R2.inverse()*y2_QR;
    cout << "The solution x2 is:" << endl << x2_QR << endl;

    double QR_rel_err2 = rel_err(x2_QR,x);
    cout << "The relative error is: \n" << QR_rel_err2 << endl;

    cout << "\n";

    MatrixXd Q3, R3;

    QR_decomposition(A3, Q3, R3);

    //cout << "Matrice Q3:" << endl << Q3 << endl;
    //cout << "Matrice R3:" << endl << R3 << endl;

    VectorXd y3_QR = Q3.transpose() * b3;

    VectorXd x3_QR = R3.inverse()*y3_QR;
    cout << "The solution x3 is:" << endl << x3_QR << endl;

    double QR_rel_err3 = rel_err(x3_QR,x);
    cout << "The relative error is: \n" << QR_rel_err3 << endl;

    return 0;
}


/*const double eps = 1e-16;

// costruisco la funzione che calcola l'erorre relativo
double rel_err(const VectorXd& w, const VectorXd& z ) {
    double norm_w = w.norm();
    double norm_z = z.norm();

    double norm_diff = (w - z).norm();

    // calcolo l'errore relativo
    double rell_err = norm_diff/ norm_z;

    return rell_err;
}

//costruisco una funzione in grado di scambiare le righe di una matrice
void swapRows(vector<vector<double>>& A, int row1, int row2) {
    vector<double> temp = A[row1];
    A[row1] = A[row2];
    A[row2] = temp;
}
// dove l'espressione "vector<vector<double>>" rappresenta una matrice bidimensionale (o array bidimensionale) di numeri in virgola mobile

// scrivo la funzione per effettuare la decomposizione PALU
void PALU_decomposition(const vector<vector<double>>& A, vector<vector<double>>& L, vector<vector<double>>& U, vector<vector<double>>& P) {
    int n = A.size();
    L = vector<vector<double>>(n, vector<double>(n, 0.0));
    U = A;
    P = vector<vector<double>>(n, vector<double>(n, 0.0));
    for (int i = 0; i < n; i++) {
        P[i][i] = 1.0;
    }

    for (int i = 0; i < n; i++) {
        int pivot_row = i;
        for (int j = i + 1; j < n; j++) {
            if (abs(U[j][i]) > abs(U[pivot_row][i])) {
                pivot_row = j;
            }
        }
        if (abs(U[pivot_row][i]) < eps) {
            cout << "La matrice è singolare." << endl;
            return;
        }
        swapRows(U, i, pivot_row);
        swapRows(P, i, pivot_row);
        for (int j = i + 1; j < n; j++) {
            double division = U[j][i] / U[i][i];
            L[j][i] = division;
            for (int k = i; k < n; k++) {
                U[j][k] -= division * U[i][k];
            }
        }
    }
    for (int i = 0; i < n; i++) {
        L[i][i] = 1.0;
    }
}

//funzione per stampare una matrice ( all'interno del mio codice inizialmente
//ho preferito scrivere le matrici come due vettori e questo ha comportato
//un po' di problemi, ad esempio non riuscivo a stampare le matrici A1, A2, A3)
void printMatrix(const vector<vector<double>>& matrix) {
    int n = matrix.size();
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            cout << matrix[i][j] << " ";
        }
        cout << endl;
    }
}

//funzione per convetire una matrice dal formato vectorvector a Matrix
MatrixXd vectorVectorToMatrixXd(const vector<vector<double>>& matr) {
    int rows = matr.size();
    int cols = matr[0].size();

    MatrixXd result(rows, cols);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result(i, j) = matr[i][j];
        }
    }

    return result;
}

// scrivo la funzione per effettuare la decomposizione QR
void QR_decomposition(const MatrixXd& A, MatrixXd& Q, MatrixXd& R) {
    int n = A.rows();

    // calcolo la fattorizzazione QR utilizzando la classe HouseholderQR
    HouseholderQR<MatrixXd> qr(A);

    Q = qr.householderQ();
    R = qr.matrixQR().triangularView<Upper>();

    // normalizzo la matrice Q (le colonne di Q sono già ortonormali)
    for (int i = 0; i < n; ++i) {
        double norma_colonna = Q.col(i).norm();
        Q.col(i) /= norma_colonna;
        R.row(i) *= norma_colonna;
    }
}


int main()
{
  //definisco la soluzione comune ai tre sistemi
  Vector2d x = Vector2d:: Ones();
  x << -1.0e+0, -1.0e+00;
  //cout << "the solution of the three systems is:" << endl;
  //cout << x << endl;

  //descrivo A e b dei tre sistemi lineari Ax=b
  //sistema 1
  //MatrixXd A1 = MatrixXd::Zero(2, 2);

  vector<vector<double>> A1 = {
      {5.547001962252291e-01,-3.770900990025203e-02},
      {8.320502943378437e-01, -9.992887623566787e-01}
  };

  //cout << "A1 matrix:" << endl;
  //printMatrix(A1);

  Vector2d b1 = Vector2d::Ones();
  b1 << -5.169911863249772e-01, 1.672384680188350e-01;

  //cout << "b1 (column vector):" << endl;
  //cout << b1 << endl;

  //sistema 2
  vector<vector<double>> A2 = {
      {5.547001962252291e-01,-5.540607316466765e-01},
      {8.320502943378437e-01,-8.324762492991313e-01}
  };

  //cout << "A2 matrix:" << endl;
  //printMatrix(A2);

  Vector2d b2 = Vector2d::Ones();
  b2 << -6.394645785530173e-04, 4.259549612877223e-04;

  //cout << "b2 (column vector):" << endl;
  //cout << b2 << endl;

  //sistema 3
  vector<vector<double>> A3 = {
      {5.547001962252291e-01,-5.547001955851905e-01},
      {8.320502943378437e-01,-8.320502947645361e-01}
  };

  //cout << "A3 matrix:" << endl;
  //printMatrix(A3);

  Vector2d b3 = Vector2d::Ones();
  b3 << -6.400391328043042e-10,  4.266924591433963e-10;

  //cout << "b3 (column vector):" << endl;
  //cout << b3 << endl;

  //applico la fattorizzazione PALU ai tre sistemi
  vector<vector<double>> L, U, P;

  cout << "Fattorizzazione PALU";
  cout << "\n" << endl;

  PALU_decomposition(A1, L, U, P);

  cout << "Matrice L1:" << endl;
  printMatrix(L);

  cout << "Matrice U1:" << endl;
  printMatrix(U);

  cout << "Matrice P1:" << endl;
  printMatrix(P);

  //trasformo le matrici in MatrixXd
  MatrixXd P1 = vectorVectorToMatrixXd(P);
  MatrixXd L1 = vectorVectorToMatrixXd(L);
  MatrixXd U1 = vectorVectorToMatrixXd(U);
  VectorXd x1 = P1.transpose() * L1.inverse() * U1.inverse() * b1;
  cout << "x1 is: \n" << x1 << endl;

  double PALU_rel_err1 = rel_err(x1,x);
  cout << "The relative error is: \n" << PALU_rel_err1 << endl;

  cout << "\n" << endl;

  PALU_decomposition(A2, L, U, P);

  cout << "Matrice L2:" << endl;
  printMatrix(L);

  cout << "Matrice U2:" << endl;
  printMatrix(U);

  cout << "Matrice P2:" << endl;
  printMatrix(P);

  MatrixXd P2 = vectorVectorToMatrixXd(P);
  MatrixXd L2 = vectorVectorToMatrixXd(L);
  MatrixXd U2 = vectorVectorToMatrixXd(U);
  VectorXd x2 = P2.transpose() * L2.inverse() * U2.inverse() * b2;
  cout << "x2 is: \n" << x2 << endl;

  double PALU_rel_err2 = rel_err(x2,x);
  cout << "The relative error is: \n" << PALU_rel_err2 << endl;

  cout << "\n" << endl;

  PALU_decomposition(A3, L, U, P);

  cout << "Matrice L3:" << endl;
  printMatrix(L);

  cout << "Matrice U3:" << endl;
  printMatrix(U);

  cout << "Matrice P3:" << endl;
  printMatrix(P);

  MatrixXd P3 = vectorVectorToMatrixXd(P);
  MatrixXd L3 = vectorVectorToMatrixXd(L);
  MatrixXd U3 = vectorVectorToMatrixXd(U);
  VectorXd x3 = P3.transpose() * L3.inverse() * U3.inverse() * b3;
  cout << "x3 is: \n" << x3 << endl;

  double PALU_rel_err3 = rel_err(x3,x);
  cout << "The relative error is: \n" << PALU_rel_err3 << endl;

  cout << "\n" << endl;

  //applico la fattorizzazione QR ai tre sistemi
  cout << "Fattorizzazione QR" << endl << "\n" << endl;
  // converto la matrice A1 (e poi A2, A3) da vectorvector a Matrix
  MatrixXd A1_QR = vectorVectorToMatrixXd(A1);

  MatrixXd Q1, R1;

  // calcolo la fattorizzazione QR
  QR_decomposition(A1_QR, Q1, R1);

  // stampo le matrici Q e R
  cout << "Matrice Q1:" << endl << Q1 << endl;
  cout << "Matrice R1:" << endl << R1 << endl;

  // calcolo il vettore y = Q^Tb
  VectorXd y1_QR = Q1.transpose() * b1;

  // risolvo il sistema Rx = y per ottenere il vettore soluzione x
  VectorXd x1_QR = R1.inverse()*y1_QR;
  cout << "x1 is:" << endl << x1_QR << endl;

  double QR_rel_err1 = rel_err(x1_QR,x);
  cout << "The relative error is: \n" << QR_rel_err1 << endl;

  cout << "\n";

  MatrixXd A2_QR = vectorVectorToMatrixXd(A2);

  MatrixXd Q2, R2;

  QR_decomposition(A2_QR, Q2, R2);

  cout << "Matrice Q2:" << endl << Q2 << endl;
  cout << "Matrice R2:" << endl << R2 << endl;

  VectorXd y2_QR = Q2.transpose() * b2;

  VectorXd x2_QR = R2.inverse()*y2_QR;
  cout << "x2 is:" << endl << x2_QR << endl;

  double QR_rel_err2 = rel_err(x2_QR,x);
  cout << "The relative error is: \n" << QR_rel_err2 << endl;

  cout << "\n";

  MatrixXd A3_QR = vectorVectorToMatrixXd(A3);

  MatrixXd Q3, R3;

  QR_decomposition(A3_QR, Q3, R3);

  cout << "Matrice Q3:" << endl << Q3 << endl;
  cout << "Matrice R3:" << endl << R3 << endl;

  VectorXd y3_QR = Q3.transpose() * b3;

  VectorXd x3_QR = R3.inverse()*y3_QR;
  cout << "x3 is:" << endl << x3_QR << endl;

  double QR_rel_err3 = rel_err(x3_QR,x);
  cout << "The relative error is: \n" << QR_rel_err3 << endl;

  return 0;
}*/
