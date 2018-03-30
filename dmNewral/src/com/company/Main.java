package com.company;

import java.io.*;
import java.util.Scanner;

public class Main {

    public static void main(String[] args) throws IOException {
        int data = 260;
        int nodeInLay = 19;
        int nodeHidLay = 11;
        int nodeOutLay = 2;
        double[] biasHid = new double[nodeHidLay];


        double[][] inputIH = new double[data][nodeInLay];

        File file = new File("input.txt");
        Scanner sc = new Scanner(file);

        while (sc.hasNextDouble()) {
            for (int i = 0; i < inputIH.length; i++) {
                for (int j = 0; j < inputIH[i].length; j++) {
                    inputIH[i][j] = sc.nextDouble();
                }
            }
        }
        System.out.println("Input = ");
        printMatrix(inputIH);
        System.out.println();

        double[][] IHWeight = new double[nodeInLay][nodeHidLay];

        for (int i = 0; i < nodeInLay; i++) {
            for (int j = 0; j < nodeHidLay; j++) {
                IHWeight[i][j] = -1+Math.random()*3;
            }
        }
        System.out.println("Input-Hidden weight = ");
        printMatrix(IHWeight);
        System.out.println();

        double[][] inputHO = multiplyMat(inputIH, IHWeight);
        System.out.println("Input Hidden-Output = ");
        printMatrix(inputHO);
        System.out.println();
        double[][] HOWeight = new double[nodeHidLay][nodeOutLay];

        for (int i = 0; i < nodeHidLay; i++) {
            for (int j = 0; j < nodeOutLay; j++) {
                HOWeight[i][j] = -1+Math.random()*3;
            }
        }
        System.out.println("Hidden-Output weight = ");
        printMatrix(HOWeight);
        System.out.println();

        double[][] out = multiplyMat(inputHO, HOWeight);
        System.out.println("Output = ");
        printMatrix(out);


    }

    private static void printMatrix(double[][] mat) {
        for (int i = 0; i < mat.length; i++) {
            // Loop through all elements of current row
            for (int j = 0; j < mat[i].length; j++) {
                System.out.print(mat[i][j] + " ");

            }
            System.out.println();
        }
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
}
