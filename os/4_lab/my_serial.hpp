#if defined (WIN32)
#	include <Windows.h>        // HANDLE и все функции read/write
#	define MY_PORT_HANDLE      HANDLE
#	define MY_PORT_SETTINGS    DCB
#	define MY_INVALID_HANDLE   INVALID_HANDLE_VALUE
#else
#	include <termios.h>      // настройки серийного порта в POSIX
#	include <unistd.h>		 // close
#	include <sys/ioctl.h>	 // ioctl
#	include <fcntl.h>		 // open, O_RDWR
#	include <errno.h>        // errno
#	define MY_PORT_HANDLE      int32_t
#	define MY_PORT_SETTINGS    termios
#	define MY_INVALID_HANDLE   -1
#endif

#include <string>    // std::string
#include <cstring>   // strcmp()
#include <cstdint>

#define MY_PORT_READ_BUF	1500
#define MY_PORT_WRITE_BUF   1500
#define SERIAL_PORT_DEFAULT_TIMEOUT			   1.0

namespace cplib
{
	class SerialPort
	{
	public:
		// Скорости
		enum BaudRate
		{
#if defined (WIN32)
			BAUDRATE_4800				= CBR_4800,		// 4800 bps
			BAUDRATE_9600				= CBR_9600,		// 9600 bps
			BAUDRATE_19200				= CBR_19200,	// 19200 bps
			BAUDRATE_38400				= CBR_38400,	// 38400 bps
			BAUDRATE_57600				= CBR_57600,	// 57600 bps
			BAUDRATE_115200				= CBR_115200,	// 115200 bps
#else
			BAUDRATE_4800 = B4800,		// 4800 bps
			BAUDRATE_9600 = B9600,		// 9600 bps
			BAUDRATE_19200 = B19200,	    // 19200 bps
			BAUDRATE_38400 = B38400,	    // 38400 bps
			BAUDRATE_57600 = B57600,	    // 57600 bps
			BAUDRATE_115200 = B115200,	    // 115200 bps
#endif
			BAUDRATE_INVALID            = -1
		};

		// Parity (четность)
		enum Parity
		{
#if defined (WIN32)
			COM_PARITY_NONE = NOPARITY,
			COM_PARITY_ODD = ODDPARITY,
			COM_PARITY_EVEN = EVENPARITY,
			COM_PARITY_MARK = MARKPARITY,
			COM_PARITY_SPACE = SPACEPARITY
#else
			COM_PARITY_NONE,
			COM_PARITY_ODD,
			COM_PARITY_EVEN,
			// в POSIX нет MARK и SPACE parity
#	if defined (_BSD_SOURCE) || defined (_SVID_SOURCE)
			COM_PARITY_MARK,
			COM_PARITY_SPACE
#	endif
#endif
		};

		// Стоповые биты
		enum StopBits
		{
#if defined (WIN32)
			STOPBIT_ONE = ONESTOPBIT,
			STOPBIT_TWO = TWOSTOPBITS
#else
			STOPBIT_ONE,
			STOPBIT_TWO
#endif
		};

		// Контроль потока
		enum FlowControl
		{
			CONTROL_NONE = 0,
#if defined (WIN32) || defined (_BSD_SOURCE) || defined (_SVID_SOURCE) || defined (__QNXNTO__)
			CONTROL_HARDWARE_RTS_CTS = 0x01,
#endif
			// Контроль DSR/DTR не поддерживается в POSIX
#if defined (WIN32) 
			CONTROL_HARDWARE_DSR_DTR = 0x02,
#endif
			CONTROL_SOFTWARE_XON_IN = 0x04,
			CONTROL_SOFTWARE_XON_OUT = 0x08
		};

		// Коды ошибок класса
		enum ErrorCodes
		{
			RE_OK = 0,
			RE_PORT_CONNECTED,
			RE_PORT_CONNECTION_FAILED,
			RE_PORT_INVALID_SETTINGS,
			RE_PORT_PARAMETERS_SET_FAILED,
			RE_PORT_PARAMETERS_GET_FAILED,
			RE_PORT_NOT_CONNECTED,
			RE_PORT_SYSTEM_ERROR,
			RE_PORT_WRITE_FAILED,
			RE_PORT_READ_FAILED
		};

		// Параметры серийного порта
		struct Parameters
		{
			// Стандартные параметры с установкой скорости
			Parameters(BaudRate speed = BAUDRATE_INVALID) {
				Defaults();
				if (speed != BAUDRATE_INVALID)
					baud_rate = speed;
			}
			// Стандартные параметры + скорость строкой
			Parameters(const char* speed) {
				Defaults();
				baud_rate = BaudrateFromString(speed);
			}
			// Строка --> baudrate
			static BaudRate BaudrateFromString(const char* baud) {
				if (!strcmp(baud,"4800"))
					return BAUDRATE_4800;
				if (!strcmp(baud,"9600"))
					return BAUDRATE_9600;
				if (!strcmp(baud,"19200"))
					return BAUDRATE_19200;
				if (!strcmp(baud,"38400"))
					return BAUDRATE_38400;
				if (!strcmp(baud,"57600"))
					return BAUDRATE_57600;
				if (!strcmp(baud,"115200"))
					return BAUDRATE_115200;
				return BAUDRATE_INVALID;
			}
			// baudrate --> строка
			static const char* StringFromBaudrate(BaudRate baud) {
				switch(baud) {
					case BAUDRATE_4800: return "4800";
					case BAUDRATE_9600: return "9600";
					case BAUDRATE_19200: return "19200";
					case BAUDRATE_38400: return "38400";
					case BAUDRATE_57600: return "57600";
					case BAUDRATE_115200: return "115200";
					default: return NULL;
				}
				return NULL;
			}
			// Дефолтные настройки
			void Defaults()
			{
				baud_rate         = BAUDRATE_115200; 
				stop_bits         = STOPBIT_ONE;
				parity            = COM_PARITY_NONE;
				controls          = CONTROL_NONE; 
				data_bits         = 8; 
				timeout           = 0.0;
				read_buffer_size  = MY_PORT_READ_BUF;
				write_buffer_size = MY_PORT_WRITE_BUF;
				on_char           = 0; 
				off_char          = (unsigned char)0xFF;
				xon_lim           = 128; 
				xoff_lim          = 128;
			}
			bool IsValid() const {
				return (baud_rate != BAUDRATE_INVALID);
			}
			
			BaudRate         baud_rate; 
			StopBits         stop_bits; 
			Parity           parity;
			int              controls; 
			unsigned char    data_bits; 
			double           timeout;
			size_t           read_buffer_size; 
			size_t           write_buffer_size;
			unsigned char    on_char; 
			unsigned char    off_char;
			int              xon_lim; 
			int              xoff_lim;
		};
	private:
		// Открыть порт
		int CreatePortHandle(const std::string& name) {
#if defined(WIN32)
			std::string sys_name = "\\\\.\\" + name;
			// Создаем handler
			// Используем CreateFileA, а не CreateFile, так как у COM-портов могут быть только ANSI-имена
			_phandle = CreateFileA(
				sys_name.c_str(),                // имя порта
				GENERIC_READ | GENERIC_WRITE,    // будем и читать и писать
				0,                               // share mode (0 для сериныйх портоа)
				NULL,                            // security attributes (NULL для серийных портов)
				OPEN_EXISTING,                   // как создавать (OPEN_EXISTING для серийных портов) 
				0,                               // файловые флаги и атрибуты(0 для серийных портов) 
				NULL);                           // файл шаблона (NULL для серийных портов)
			if (_phandle == MY_INVALID_HANDLE)
				return RE_PORT_CONNECTION_FAILED;
#else
			// Открыть порт
			// O_RDWR: Read+Write
			// O_NOCTTY: сырой ввод, без управляющего терминала
			_phandle = ::open(name.c_str(), O_RDWR | O_NOCTTY);
			if (_phandle < 0) {
				_phandle = MY_INVALID_HANDLE;
				return RE_PORT_CONNECTION_FAILED;
			}
#endif
			return RE_OK;
		}

		// Закрыть порт
		int ClosePortHandle() {
			int ret = RE_OK;
#if defined(WIN32)
			if (!::CloseHandle(_phandle))
				ret = RE_PORT_SYSTEM_ERROR;
#else
			tcflush(_phandle, TCIOFLUSH);
			if (::close(_phandle) < 0)
				ret = RE_PORT_SYSTEM_ERROR;
#endif
			_phandle = MY_INVALID_HANDLE;
			return ret;
		}
		
		// Параметры класса - в параметры порта системы
		int ParamsToSystem(const Parameters& inp_params, MY_PORT_SETTINGS& params) {
			// заполним параметры
			memset(&params,0,sizeof(params));
			if (!inp_params.IsValid())
				return RE_PORT_INVALID_SETTINGS;
#if defined(WIN32)
			params.DCBlength = sizeof(DCB);                 // длина структуры
			GetCommState(_phandle, &params);                // получим текущее состояние настроек порта
			params.fBinary = TRUE;                          // Windows поддердивает только бинарный режим
			params.BaudRate = DWORD(inp_params.baud_rate);  // скорость порта
			params.ByteSize = BYTE(inp_params.data_bits);   // длина слова
			params.Parity   = BYTE(inp_params.parity);      // четность
			params.StopBits = BYTE(inp_params.stop_bits);   // число стоповых бит
			params.fAbortOnError = FALSE;                   // на ошибке выходить не будем
			// CTS (clear-to-send)\ RTS (request-to-send)
			if (inp_params.controls & CONTROL_HARDWARE_RTS_CTS) {
				params.fOutxCtsFlow = TRUE;
				params.fRtsControl = RTS_CONTROL_ENABLE;
			}
			else {
				params.fOutxCtsFlow =  FALSE;
				params.fRtsControl = RTS_CONTROL_DISABLE;
			}
			// DSR (data-set-ready) \ DTR (data-terminal-ready)
			if (inp_params.controls & CONTROL_HARDWARE_DSR_DTR) {
				params.fOutxDsrFlow = TRUE;
				params.fDsrSensitivity = TRUE;
				params.fDtrControl = DTR_CONTROL_ENABLE;
			}
			else {
				params.fOutxDsrFlow = FALSE;
				params.fDsrSensitivity = FALSE;
				params.fDtrControl = DTR_CONTROL_DISABLE;
			}
			// XON/XOFF control
			if (inp_params.controls & CONTROL_SOFTWARE_XON_IN)
				params.fInX = TRUE;
			else
				params.fInX = FALSE;
			if (inp_params.controls & CONTROL_SOFTWARE_XON_OUT)
				params.fOutX = TRUE;
			else
				params.fOutX = FALSE;
			params.XonChar = inp_params.on_char;
			params.XoffChar = inp_params.off_char;
			params.XonLim = inp_params.xon_lim;
			params.XoffLim = inp_params.xoff_lim;
		
			// Прочие параметры
			params.fErrorChar = FALSE;
			params.fNull = FALSE;
#else
			// получим текущее состояние настроек порта
			if (tcgetattr(_phandle, &params) != 0)
				return RE_PORT_PARAMETERS_SET_FAILED;
			// Установим скорость
			if (cfsetispeed(&params, inp_params.baud_rate) != 0)
				return RE_PORT_PARAMETERS_SET_FAILED;
			if (cfsetospeed(&params, inp_params.baud_rate) != 0)
				return RE_PORT_PARAMETERS_SET_FAILED;
			// длина слова
			params.c_cflag &= ~CSIZE; // character size mask
			int flg;
			if(inp_params.data_bits == 5)
				flg = CS5;
			else if (inp_params.data_bits == 6)
				flg = CS6;
			else if (inp_params.data_bits == 7)
				flg = CS7;
			else
				flg = CS8;
			params.c_cflag |= flg;
			// четность
			if (inp_params.parity == COM_PARITY_NONE)
				params.c_cflag &= ~PARENB;
			else
				params.c_cflag |= PARENB;
			if (inp_params.parity == COM_PARITY_ODD)
				params.c_cflag |= PARODD;
			else
				params.c_cflag &= ~PARODD;
#if defined (_BSD_SOURCE) || defined (_SVID_SOURCE)
			if (inp_params.parity == COM_PARITY_MARK)
				params.c_cflag |= (CMSPAR | PARODD);
			else if (inp_params.parity == COM_PARITY_SPACE)
				params.c_cflag |= CMSPAR;
			else
				params.c_cflag &= ~CMSPAR;
#endif
			// число стоповых бит
			if (inp_params.stop_bits == STOPBIT_TWO)
				params.c_cflag |= CSTOPB;
			else
				params.c_cflag &= ~CSTOPB;
			// CTS
#if defined (_BSD_SOURCE) || defined (_SVID_SOURCE)
			if (inp_params.controls & CONTROL_HARDWARE_RTS_CTS)
				params.c_cflag |= CRTSCTS;
			else
				params.c_cflag &= ~CRTSCTS;
#endif
			// программное управление потоком
			if (inp_params.controls & CONTROL_SOFTWARE_XON_OUT)
				params.c_iflag |= IXON;
			else
				params.c_iflag &= ~IXON;
			if (inp_params.controls & CONTROL_SOFTWARE_XON_IN)
				params.c_iflag |= IXOFF; 
			else
				params.c_iflag &= ~IXOFF;
			if (!((inp_params.controls & CONTROL_SOFTWARE_XON_OUT) || 
				  (inp_params.controls & CONTROL_SOFTWARE_XON_IN)))
				params.c_iflag &= ~(IGNBRK|BRKINT|PARMRK|ISTRIP|INLCR|IGNCR|ICRNL);

			// Таймаут
			params.c_cc[VMIN]     = 0; // Минимальное число байт для чтения от устройства
			params.c_cc[VTIME]    = ((int32_t)(inp_params.timeout*1e3)+99)/100;

			// Включим получение данных и установим локальный режим
			params.c_cflag |= (CLOCAL | CREAD);
			// Отключим эхо
			params.c_lflag &= ~(ECHO|ECHONL|ICANON|ISIG|IEXTEN);
			// установим сырой вывод
			params.c_oflag &= ~OPOST;
#endif
			return RE_OK;
		}

		// Установить параметры
		int SetParameters(const Parameters& inp_params) {
			if (!IsOpen())
				return RE_PORT_NOT_CONNECTED;
			// Сконвертируем параметры класса в системные параметры COM-порта
			MY_PORT_SETTINGS setts;
			int ret = ParamsToSystem(inp_params, setts);
			if (!ret)
				return ret;
#if defined(WIN32)
			// Системный вызов установки параметров
			if(!SetCommState(_phandle, &setts))
				return RE_PORT_PARAMETERS_SET_FAILED;
			// В Windows необходимо установить размеры буферов отправки\получения
			if (!SetupComm(_phandle,(DWORD)inp_params.read_buffer_size, (DWORD)inp_params.write_buffer_size))
				return RE_PORT_PARAMETERS_SET_FAILED;
			// Также нужно отдельно установить таймауты чтения\записи
			ret = SetTimeout(inp_params.timeout);
#else
			// Установим системные параметры
			if (tcsetattr(_phandle,TCSANOW, &setts))
				return RE_PORT_PARAMETERS_SET_FAILED;
			ret = RE_OK;		
#endif
			if (ret == RE_OK)
				_timeout = inp_params.timeout;
			return ret;
		}

	public:
		// Конструктор по-умолчанию
		SerialPort():_phandle(MY_INVALID_HANDLE),_timeout(0.0){}
		SerialPort(const std::string& name, BaudRate speed):_phandle(MY_INVALID_HANDLE), _timeout(0.0) {
			Open(name,Parameters(speed));
		}
		// Деструктор
		virtual ~SerialPort() {
			if (IsOpen()) 
				Close();
		}
		// Открыть серийный порт 
		int Open(const std::string& port_name, const Parameters& params) {
			if (IsOpen())
				return RE_PORT_CONNECTED;
			int ret = CreatePortHandle(port_name);
			if (ret != RE_OK)
				return ret;
			_port_name = port_name;
			ret = SetParameters(params);
			if (ret != RE_OK)
				Close();
			return ret;
		}
		// Закрыть серийный порт
		int Close() {
			if (!IsOpen())
				return RE_PORT_NOT_CONNECTED;
			int ret = ClosePortHandle();
			_timeout = 0.0;
			_port_name.clear();
			return ret;
		}
		// Открыт ли порт
		bool IsOpen() const {
			return (_phandle != MY_INVALID_HANDLE);
		}
		// Имя порта
		const std::string& GetPortName() {
			return _port_name;
		}
		// Установленный таймаут операций
		double GetTimeout() {
			return _timeout;
		}
		// Установить таймаут для блокирующих операций
		int SetTimeout(double timeout) {
			if (!IsOpen())
				return RE_PORT_NOT_CONNECTED;
			int ret = RE_OK;
			int tmms = (int32_t)(timeout*1e3);
#if defined(WIN32)
			// Set COM timeouts
			COMMTIMEOUTS tmts;
			if (!GetCommTimeouts(_phandle,&tmts))
				return RE_PORT_PARAMETERS_GET_FAILED;
			// Если таймаут нулевой - функция должна выходить мгновенно, даже если данных нет
			// В противном случае - ждем заданное время
			if (tmms > 0)
				tmts.ReadIntervalTimeout = 0;
			else
				tmts.ReadIntervalTimeout = MAXDWORD;
			tmts.ReadTotalTimeoutConstant = tmms;
			tmts.ReadTotalTimeoutMultiplier = 0;
			tmts.WriteTotalTimeoutConstant = tmms;
			tmts.WriteTotalTimeoutMultiplier = 0;
			if (!SetCommTimeouts(_phandle, &tmts))
				return RE_PORT_PARAMETERS_SET_FAILED;
#else
			MY_PORT_SETTINGS params;
			if (tcgetattr(_phandle, &params))
				return RE_PORT_PARAMETERS_GET_FAILED;
			// Читаем минимально 0 байт, таймаут - в децисекундах (0.1 секунды)
			params.c_cc[VMIN]     = 0;
			params.c_cc[VTIME]    = (tmms+99)/100;
			// Установим системные параметры
			if (tcsetattr(_phandle,TCSANOW, &params))
				return RE_PORT_PARAMETERS_SET_FAILED;
#endif
			if (ret == RE_OK)
				_timeout = timeout;
			return ret;
		}
		// Пишем буфер указанного размера в порт
		// Функция возвращает код ошибки
		int Write (const void* buf, size_t buf_size, size_t* written = NULL) {
			if (!IsOpen())
				return RE_PORT_NOT_CONNECTED;
			if (written)
				*written = 0;
#ifdef _WIN32
			DWORD feedback = 0;
			if (! ::WriteFile(_phandle, (LPCVOID)buf, (DWORD)buf_size, &feedback, NULL))
				return RE_PORT_WRITE_FAILED;
			if (written)
				*written = feedback;
#else
			size_t result = write(_phandle, buf, buf_size);
			if (result < 0)
				return RE_PORT_WRITE_FAILED;
			if (written)
				*written = result;
#endif
			return RE_OK;
		}
		// Пишем строку в порт, \0 не пишем
		int Write (const std::string& data) {
			size_t wrt = 0;
			int ret = RE_OK;
			for (size_t i = 0; i < data.size(); i+=wrt) {
				ret = Write(data.c_str()+i,data.size()-i,&wrt);
				if (ret != RE_OK)
					break;
			}
			return ret;
		}
		// Читаем из порта данные в буфер размера max_size
		// Таймаут чтения задается через SetTimeout()
		int Read(void* buf, size_t max_size, size_t* readd) {
			if (!IsOpen()) {
				return RE_PORT_NOT_CONNECTED;
			}
			*readd = 0;
#ifdef _WIN32
			DWORD feedback = 0;
			if (! ::ReadFile(_phandle, buf, (DWORD)max_size, &feedback, NULL))
				return RE_PORT_READ_FAILED;
			*readd = (size_t)feedback;
#else
			int res = read(_phandle, buf, max_size);
			if (res < 0)
				return RE_PORT_READ_FAILED;
			*readd = (size_t)res;
#endif
			return RE_OK;
		}
		// Читаем из порта строку
		// Ограничение чтения - таймаут, либо символ \n
		int Read(std::string& str, double timeout = SERIAL_PORT_DEFAULT_TIMEOUT) {
			int ret = RE_OK;
			str.reserve(255);
			size_t rd = 0;
			ret = Read((void*)str.c_str(),str.capacity(),&rd);
            if (ret != RE_OK)
                return ret;
			str.resize(rd);
			return ret;
		}

		// Отправить все ожидающие данные устройству
		int Flush() {
			if (!IsOpen())
				return RE_PORT_NOT_CONNECTED;
#if defined(WIN32)
			// Remove any 'old' data in buffer
			if (!PurgeComm(_phandle, PURGE_TXCLEAR | PURGE_RXCLEAR))
				return RE_PORT_SYSTEM_ERROR;
#else
			tcflush(_phandle, TCIOFLUSH);
#endif
			return RE_OK;
		}

		// Операторы для удобного чтения\записи строк
		SerialPort& operator<< (const std::string& data) {Write(data); return *this;}
		SerialPort& operator>> (std::string& data) {Read(data); return *this;}

	private:

		MY_PORT_HANDLE _phandle;
		std::string    _port_name;
		double         _timeout;
		
	private:
		// Защита от копирования
		SerialPort(const SerialPort& port){}
		SerialPort& operator= (const SerialPort& port){return *this;}
	};
}
