#include "mainwindow.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QMessageBox>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
{
    client = new qtwebclient(this);
    QWidget *central = new QWidget(this);
    setCentralWidget(central);

    currentLabel = new QLabel("Текущая температура: -- С");

    fromEdit = new QDateTimeEdit(QDateTime::currentDateTime().addDays(-1));
    toEdit = new QDateTimeEdit(QDateTime::currentDateTime());

    fromEdit->setDisplayFormat("yyyy-MM-dd HH:mm");
    toEdit->setDisplayFormat("yyyy-MM-dd HH:mm");

    fromEdit->setCalendarPopup(true);
    toEdit->setCalendarPopup(true);

    requestButton = new QPushButton("Построить график");

    plot = new QwtPlot();
    plot->setTitle("Температура за период");
    plot->setAxisTitle(QwtPlot::xBottom, "Время");
    plot->setAxisTitle(QwtPlot::yLeft, "Температура");
    plot->setCanvasBackground(Qt::white);

    curve = new QwtPlotCurve("Температура");
    curve->setPen(Qt::blue, 2);
    curve->attach(plot);

    QHBoxLayout *controlsLayout = new QHBoxLayout();
    controlsLayout->addWidget(fromEdit);
    controlsLayout->addWidget(toEdit);
    controlsLayout->addWidget(requestButton);

    QVBoxLayout *mainLayout = new QVBoxLayout();
    mainLayout->addWidget(currentLabel);
    mainLayout->addLayout(controlsLayout);
    mainLayout->addWidget(plot);

    central->setLayout(mainLayout);

    connect(requestButton, &QPushButton::clicked,
            this, &MainWindow::onRequestClicked);

    connect(client, &qtwebclient::currentReceived,
            this, &MainWindow::onCurrentReceived);

    connect(client, &qtwebclient::statsReceived,
            this, &MainWindow::onStatsReceived);

    connect(client, &qtwebclient::errorOccurred,
            this, &MainWindow::onError);

    client->requestCurrent();

}

void MainWindow::onRequestClicked()
{
    qint64 from = fromEdit->dateTime().toSecsSinceEpoch();
    qint64 to   = toEdit->dateTime().toSecsSinceEpoch();

    if (from >= to) {
        QMessageBox::warning(this, "Ошибка", "Дата FROM должна быть меньше TO");
        return;
    }

    client->requestStats(from, to);
}

void MainWindow::onCurrentReceived(double value)
{
    currentLabel->setText(
        QString("Текущая температура: %1 °C").arg(value, 0, 'f', 1)
        );
}

void MainWindow::onStatsReceived(QVector<double> values, QVector<qint64> timestamps)
{
    QVector<double> x(values.size());
    QVector<double> y = values;

    qint64 t0 = timestamps.first();
    for (int i = 0; i < timestamps.size(); ++i) {
        x[i] = (timestamps[i] - t0) / 3600.0; // часы
    }

    curve->setSamples(x, y);
    plot->replot();
}

void MainWindow::onError(QString msg)
{
    QMessageBox::critical(this, "Ошибка: ", msg);
}

MainWindow::~MainWindow() {}
