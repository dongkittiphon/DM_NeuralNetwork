package com.company;

import java.io.*;
import java.util.Scanner;


public class Main {

    public static void main(String[] args) throws IOException {
        int data = 10;
        int nodeInLay = 19;
        int nodeHidLay = 11;
        int nodeOutLay = 2;
        double lernWeg = -0.4;

        double[][] error = new double[data][nodeOutLay];

        double[] biasHid = new double[nodeHidLay];
        for (int i = 0; i < nodeHidLay; i++) {
            biasHid[i] = -1+Math.random()*2;
        }

        double[] biasOut = new double[nodeOutLay];
        for (int i = 0; i < nodeOutLay; i++) {
            biasOut[i] = -1+Math.random()*2;
        }
        double[][] inputIH = new double[data][nodeInLay];
        double[][] output = new double[data][nodeOutLay];

        File file1 = new File("input.txt");
        File file2 = new File("output.txt");

        Scanner sc1 = new Scanner(file1);
        Scanner sc2 = new Scanner(file2);

        while (sc1.hasNextDouble()) {
            for (int i = 0; i < inputIH.length; i++) {
                for (int j = 0; j < inputIH[i].length; j++) {
                    inputIH[i][j] = sc1.nextDouble();
                }
            }
        }

        while (sc2.hasNextDouble()){
            for (int i = 0; i < output.length; i++) {
                for (int j = 0; j < output[i].length; j++) {
                    output[i][j] = sc2.nextDouble();
                }
            }
        }

        System.out.println("Input = ");
        printMatrix(inputIH);
        System.out.println();
        System.out.println("output = ");
        printMatrix(output);
        System.out.println();

        double[][] IHWeight = new double[nodeInLay][nodeHidLay];
        for (int i = 0; i < nodeInLay; i++) {
            for (int j = 0; j < nodeHidLay; j++) {
                IHWeight[i][j] = -1+Math.random()*2;
            }
        }

        double[][] HOWeight = new double[nodeHidLay][nodeOutLay];
        for (int i = 0; i < nodeHidLay; i++) {
            for (int j = 0; j < nodeOutLay; j++) {
                HOWeight[i][j] = -1+Math.random()*2;
            }
        }

        double[] err = new double[nodeOutLay];
        double[] fdivO = new double[nodeOutLay];
        double[] graDianO = new double[nodeOutLay];

        double[] fdivH = new double[nodeHidLay];
        double[] graDianH = new double[nodeHidLay];

        for (int f = 0; f < 1; f++) {
            for (int i = 0; i < data; i++) {
                System.out.println("Round: " + (i + 1));
                System.out.println("Weight(in-hid) = ");
                printMatrix(IHWeight);
                System.out.println("Bias(in-hid) = ");
                printArray(biasHid);

                double[][] inputHO = multiplyMat(inputIH, IHWeight);
                inputHO = addmat(inputHO, biasHid);
                inputHO = sigmoid(inputHO);

            System.out.println("Input(hid-out)= ");
            printMatrix(inputHO);
//                System.out.println("Weight(hid-out) = ");
//                printMatrix(HOWeight);
//                System.out.println("Bias(hid-out) = ");
//                printArray(biasOut);
                double[][] out = multiplyMat(inputHO, HOWeight);
                out = addmat(out, biasOut);
                out = sigmoid(out);

                System.out.println("Output = ");
                printMatrix(out);
                System.out.println();

                for (int j = 0; j < nodeOutLay; j++) {
                    err[j] = out[i][j] - output[i][j];
                    fdivO[j] = out[i][j] * (1 - out[i][j]);
                    error[i][j] = err[j];
                }
//                System.out.println("error1 = " + err[0]);
//                System.out.println("error2 = " + err[1]);
//                if(Math.abs(err[0]) < 0.01 && Math.abs(err[1]) < 0.01)break;
                for (int j = 0; j < nodeOutLay; j++) {
                    graDianO[j] = err[j] * fdivO[j];
                }
//
//            System.out.println("gra1O = "+graDianO[0]);
//            System.out.println("gra2O = "+graDianO[1]);
                for (int j = 0; j < nodeHidLay; j++) {
                    for (int k = 0; k < nodeOutLay; k++) {
                        HOWeight[j][k] = (lernWeg * graDianO[k] * inputHO[i][k]) + HOWeight[j][k];
                    }
                }

                double sumGra = 0;
                for (int j = 0; j < nodeOutLay; j++) {
                    sumGra += (graDianO[j] * biasOut[j]);
                }

                for (int j = 0; j < nodeOutLay; j++) {
                    biasOut[j] = (lernWeg * graDianO[j]) + biasOut[j];
                }
//                System.out.println("Weight(hid-out)ugrade = ");
//                printMatrix(HOWeight);
//                System.out.println("Bias(hid-out) upgrade= ");
//                printArray(biasOut);

                for (int j = 0; j < nodeHidLay; j++) {
                    fdivH[j] = inputHO[i][j] * (1 - inputHO[i][j]);
                    graDianH[j] = fdivH[j] * sumGra;
                }

//                 System.out.println("sumGra = " + sumGra);
//            System.out.println("gra1H = "+graDianH[0]);
//            System.out.println("gra2H = "+graDianH[1]);

                for (int j = 0; j < nodeInLay; j++) {
                    for (int k = 0; k < nodeHidLay; k++) {
                        IHWeight[j][k] = (lernWeg * graDianH[k] * inputIH[i][k]) + IHWeight[j][k];
                    }
                }
                for (int j = 0; j < nodeHidLay; j++) {
                    biasHid[j] = (lernWeg * graDianH[j]) + biasHid[j];
                }
//                System.out.println("Weight(in-hid)ugrade = ");
//                printMatrix(IHWeight);
//                System.out.println("Bias(in-hid) upgrade= ");
//                printArray(biasHid);
//                System.out.println();
            }
        }
//        printMatrix(error);
    }
    private static double[][] sigmoid(double[][] f){
        int col = 0;
        double[][] out =new double[f.length][f[col].length];

        for (int i = 0; i < f.length; i++) {
            for (int j = 0; j < f[i].length; j++) {
                out[i][j] = 1/(1+Math.exp(-f[i][j]));
            }
        }
        return out;
    }

    private static void printMatrix(double[][] mat) {
        for (int i = 0; i < mat.length; i++) {
            for (int j = 0; j < mat[i].length; j++) {
                if(mat[i][j] >= 0)System.out.print(" ");
                System.out.printf("%.4f\t", mat[i][j]);
            }
            System.out.println();
        }
    }

    private static void printArray(double[] a) {
        for (int i = 0; i < a.length; i++) {
                if(a[i] >= 0)System.out.print(" ");
                System.out.printf("%.4f\t", a[i]);
        }
        System.out.println();
    }

    private static double[][] multiplyMat(double[][] a, double[][] b) {
        int m1 = a.length;
        int n1 = a[0].length;
        int m2 = b.length;
        int n2 = b[0].length;
        if (n1 != m2) throw new RuntimeException("Illegal matrix dimensions.");
        double[][] c = new double[m1][n2];
        for (int i = 0; i < m1; i++)
            for (int j = 0; j < n2; j++)
                for (int k = 0; k < n1; k++)
                    c[i][j] += a[i][k] * b[k][j];
        return c;
    }

    private static double[][] addmat(double[][] a, double[] b) {
        int m = a.length;
        int n = a[0].length;
        double[][] c = new double[m][n];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                c[i][j] = a[i][j] + b[j];
        return c;
    }
}
