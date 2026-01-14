#pragma once
#include <QObject>
#include <QNetworkAccessManager>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>

class qtwebclient : public QObject
{
    Q_OBJECT
public:
    explicit qtwebclient(QObject *parent = nullptr);

    void requestCurrent();
    void requestStats(qint64 from, qint64 to);

signals:
    void currentReceived(double value);
    void statsReceived(QVector<double> values, QVector<qint64> timestamps);
    void errorOccurred(QString msg);

private slots:
    void onReplyFinished(QNetworkReply *reply);

private:
    QNetworkAccessManager manager;
};
