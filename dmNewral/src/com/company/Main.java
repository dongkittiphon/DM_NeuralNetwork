package com.company;

import java.io.*;
import java.util.Scanner;


public class Main {

    public static void main(String[] args) throws IOException {
        int data = 260;
        int nodeInLay = 19;
        int nodeHidLay = 11;
        int nodeOutLay = 2;
        double lernWeg = 0.3;

        double[][] error = new double[data][nodeOutLay];

        double[] biasHid = new double[nodeHidLay];
        for (int i = 0; i < nodeHidLay; i++) {
            biasHid[i] = Math.random();
        }

        double[] biasOut = new double[nodeOutLay];
        for (int i = 0; i < nodeOutLay; i++) {
            biasOut[i] = Math.random();
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
                IHWeight[i][j] = Math.random();
            }
        }

        double[][] HOWeight = new double[nodeHidLay][nodeOutLay];
        for (int i = 0; i < nodeHidLay; i++) {
            for (int j = 0; j < nodeOutLay; j++) {
                HOWeight[i][j] = Math.random();
            }
        }

        double[] err = new double[nodeOutLay];
        double[] fdivO = new double[nodeOutLay];
        double[] graDianO = new double[nodeOutLay];

        double[] fdivH = new double[nodeHidLay];
        double[] graDianH = new double[nodeHidLay];
        double[][] inputHO;
        double[][] out = new double[nodeOutLay][data];

        for (int f = 0; f < 50; f++) {
            boolean st = false;
            for (int i = 0; i < data; i++) {
//                System.out.println("Round: " + (i + 1));
//                System.out.println("Weight(in-hid) = ");
//                printMatrix(IHWeight);
//                System.out.println("Bias(in-hid) = ");
//                printArray(biasHid);

                inputHO = multiplyMat(inputIH, IHWeight);
                inputHO = addmat(inputHO, biasHid);
                inputHO = sigmoid(inputHO);


//            System.out.println("Input(hid-out)= ");
//            printMatrix(inputHO);
//                System.out.println("Weight(hid-out) = ");
//                printMatrix(HOWeight);
//                System.out.println("Bias(hid-out) = ");
//                printArray(biasOut);
                out = multiplyMat(inputHO, HOWeight);
                out = addmat(out, biasOut);
                out = sigmoid(out);

//                System.out.println("Output = ");
//                printMatrix(out);
//                System.out.println();

                //Back prop. begin

                for (int j = 0; j < nodeOutLay; j++) {
                    err[j] = output[i][j] - out[i][j];
//                    System.out.println("error["+j+"] = "+err[j]);
                }
                if(Math.abs(err[0]) < 0.001 && Math.abs(err[1]) < 0.001){
                    st = true;
                    break;
                }

                for (int j = 0; j < nodeOutLay; j++) {
//                    System.out.println("y = "+ out[i][j]);
                    fdivO[j] = out[i][j] * (1 - out[i][j]);
//                    System.out.println("fdiv = "+ fdivO[j]);
                }

                for (int j = 0; j < nodeOutLay; j++) {
                    graDianO[j] = err[j]*fdivO[j];
//                    System.out.println("gradian["+j+"] ="+graDianO[j]);
                }
//use for hiden node
                double[] sumGW = new double[nodeHidLay];
                for (int j = 0; j < sumGW.length; j++) {
                    sumGW[j] = 0;
                }

                for (int j = 0; j < nodeHidLay; j++) {
                    for (int k = 0; k < nodeOutLay; k++) {
                        sumGW[j] += graDianO[k]*HOWeight[j][k];
                    }
                }
//end use for hiden node
                for (int j = 0; j < nodeHidLay; j++) {
                    for (int k = 0; k < nodeOutLay; k++) {
//                        System.out.println("HOweight["+j+"]["+k+"] = "+HOWeight[j][k]);
                        HOWeight[j][k] = (lernWeg*graDianO[k]*inputHO[i][j])+HOWeight[j][k];
//                        System.out.println("HOweight["+j+"]["+k+"] NEW= "+HOWeight[j][k]);
//                        System.out.println("inputHO["+i+"]["+j+"] = "+inputHO[i][j]);
                    }
                }
                for (int j = 0; j < nodeOutLay; j++) {
                    biasOut[j] = (lernWeg * graDianO[j]) + biasOut[j];
                }
//hidden node begin.
                for (int j = 0; j < nodeHidLay; j++) {
                    fdivH[j] = inputHO[i][j]*(1-inputHO[i][j]);
                }

                for (int j = 0; j < nodeHidLay; j++) {
                    graDianH[j] = fdivH[j]*sumGW[j];
                }

                for (int j = 0; j < nodeInLay; j++) {
                    for (int k = 0; k < nodeHidLay; k++) {
//                        System.out.println("IHweight["+j+"]["+k+"] = "+IHWeight[j][k]);
                        IHWeight[j][k] = (lernWeg*graDianH[k]*inputIH[i][j])+IHWeight[j][k];
//                        System.out.println("IHweight["+j+"]["+k+"] NEW= "+IHWeight[j][k]);
//                        System.out.println("inputIH["+i+"]["+j+"] = "+inputIH[i][j]);
                    }
                }
                for (int j = 0; j < nodeHidLay; j++) {
                    biasHid[j] = (lernWeg * graDianH[j]) + biasHid[j];
                }




//re code start form here
                //Back prop.
//
//                for (int j = 0; j < nodeOutLay; j++) {
//                    err[j] = out[i][j] - output[i][j];
//                    fdivO[j] = out[i][j] * (1 - out[i][j]);
//                    error[i][j] = err[j];
//                }
//                System.out.println("error1 = " + err[0]);
//                System.out.println("error2 = " + err[1]);
//                System.out.println("fdiv1 = " + fdivO[0]);
//                System.out.println("fdiv2 = " + fdivO[1]);
////                if(Math.abs(err[0]) < 0.01 && Math.abs(err[1]) < 0.01)break;
//                for (int j = 0; j < nodeOutLay; j++) {
//                    graDianO[j] = err[j] * fdivO[j];
//                }
////
//            System.out.println("gra1O = "+graDianO[0]);
//            System.out.println("gra2O = "+graDianO[1]);
//
//
//                double[] sumGra = new double[nodeHidLay];
//                for (int j = 0; j < nodeHidLay; j++) {
//                    sumGra[j] = 0;
//                }
//                for (int j = 0; j < nodeHidLay; j++) {
////                    System.out.println();
//                    for (int k = 0; k < nodeOutLay; k++) {
//                        sumGra[j] += (graDianO[k] * HOWeight[j][k]);
////                        System.out.println("graO="+ graDianO[k]);
////                        System.out.println("HW weg="+ HOWeight[j][k]);
//                    }
////                    System.out.println("sum grad["+j+"]= "+ sumGra[j]);
////                    System.out.println();
//                }
//
//
//                for (int j = 0; j < nodeHidLay; j++) {
//                    for (int k = 0; k < nodeOutLay; k++) {
////                        System.out.println();
////                        System.out.println("HO weg = "+ HOWeight[j][k]);
//
//                        HOWeight[j][k] = (lernWeg * graDianO[k] * inputHO[i][k]) + HOWeight[j][k];
//
////                        System.out.println("lenweg = "+lernWeg);
////                        System.out.println("grad = "+graDianO[k]);
////                        System.out.println("input = "+ inputHO[i][k]);
////                        System.out.println("newHO weg = "+ HOWeight[j][k]);
////                        System.out.println();
//                    }
//                }
//
//
//
//                for (int j = 0; j < nodeOutLay; j++) {
//                    biasOut[j] = (lernWeg * graDianO[j]) + biasOut[j];
//                }
////                System.out.println("Weight(hid-out)ugrade = ");
////                printMatrix(HOWeight);
////                System.out.println("Bias(hid-out) upgrade= ");
////                printArray(biasOut);
//
//                for (int j = 0; j < nodeHidLay; j++) {
//                    fdivH[j] = inputHO[i][j] * (1 - inputHO[i][j]);
//                    graDianH[j] = fdivH[j] * sumGra[j];
//                }
//                System.out.println("gedianH = ");
//                printArray(graDianH);
////                 System.out.println("sumGra = " + sumGra);
////            System.out.println("gra1H = "+graDianH[0]);
////            System.out.println("gra2H = "+graDianH[1]);
//
//                for (int j = 0; j < nodeInLay; j++) {
//                    for (int k = 0; k < nodeHidLay; k++) {
//                        IHWeight[j][k] = (lernWeg * graDianH[k] * inputIH[i][k]) + IHWeight[j][k];
//                    }
//                }
//                for (int j = 0; j < nodeHidLay; j++) {
//                    biasHid[j] = (lernWeg * graDianH[j]) + biasHid[j];
//                }
////                System.out.println("Weight(in-hid)ugrade = ");
////                printMatrix(IHWeight);
////                System.out.println("Bias(in-hid) upgrade= ");
////                printArray(biasHid);
////                System.out.println();
//end from here
            }
            System.out.println("end ephoc"+(f+1));
            System.out.println("Output = ");
            printMatrix(out);
            System.out.println();
//            System.out.println("error = ");
//            printMatrix(error);
            if (st)break;
        }

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
