#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QLabel>
#include <QDateTimeEdit>
#include <QPushButton>
#include "qtwebclient.h"
#include <qwt_plot.h>
#include <qwt_plot_curve.h>
class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();
private slots:
    void onRequestClicked();
    void onCurrentReceived(double value);
    void onStatsReceived(QVector<double> values, QVector<qint64> timestamps);
    void onError(QString msg);

private:
    qtwebclient *client;

    QLabel *currentLabel;
    QDateTimeEdit *fromEdit;
    QDateTimeEdit *toEdit;
    QPushButton *requestButton;

    QwtPlot *plot;
    QwtPlotCurve *curve;
};
#endif // MAINWINDOW_H
