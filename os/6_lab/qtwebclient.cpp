#include "qtwebclient.h"

#include <QNetworkReply>
#include <QUrlQuery>

qtwebclient::qtwebclient(QObject *parent)
    : QObject(parent)
{
    connect(&manager, &QNetworkAccessManager::finished,
            this, &qtwebclient::onReplyFinished);
}

void qtwebclient::requestCurrent()
{
    QUrl url("http://127.0.0.1:8080/current");
    manager.get(QNetworkRequest(url));
}

void qtwebclient::requestStats(qint64 from, qint64 to)
{
    QUrl url("http://127.0.0.1:8080/stats");
    QUrlQuery q;
    q.addQueryItem("from", QString::number(from));
    q.addQueryItem("to", QString::number(to));
    url.setQuery(q);

    manager.get(QNetworkRequest(url));
}

void qtwebclient::onReplyFinished(QNetworkReply *reply)
{
    if (reply->error() != QNetworkReply::NoError)
    {
        emit errorOccurred(reply->errorString());
        reply->deleteLater();
        return;
    }

    QJsonDocument doc = QJsonDocument::fromJson(reply->readAll());
    QJsonObject obj = doc.object();

    if (obj.contains("temperature"))
    {
        emit currentReceived(obj["temperature"].toDouble());
    }
    else if (obj.contains("values"))
    {
        QVector<double> values;
        QVector<qint64> timestamps;

        for (auto v : obj["values"].toArray())
        {
            QJsonObject o = v.toObject();
            timestamps.push_back(o["timestamp"].toVariant().toLongLong());
            values.push_back(o["value"].toDouble());
        }

        emit statsReceived(values, timestamps);
    }

    reply->deleteLater();
}
