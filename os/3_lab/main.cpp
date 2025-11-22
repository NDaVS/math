#include <iostream>
#include <fstream>
#include <string>
#include <unistd.h>
#include <signal.h>
#include <pthread.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <mutex>
#include <semaphore.h>
#include <fcntl.h>
#include <ctime>
#include <chrono>
#include <format>

#define LOG_FILE "app.log"

struct SharedData
{
  int counter;
  pid_t leader_pid;
};

pid_t pid_copy_1 = 0;
pid_t pid_copy_2 = 0;
SharedData *shared_data = nullptr;
sem_t *shared_sem = nullptr;
const char *shm_name = "/my_shm";
const char *sem_name = "/my_sem";

std::ofstream log_fd;
std::mutex mtx;

void logMessage(const std::string &message)
{
  mtx.lock();
  log_fd << message << std::endl;
  mtx.unlock();
}

std::string getCurrentTimestamp()
{
  const auto now_time = std::chrono::system_clock::now();
  std::time_t tt = std::chrono::system_clock::to_time_t(now_time);
  std::tm tm = *std::localtime(&tt);

  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now_time.time_since_epoch()) % 1000;

  return std::format(
      "{:04}-{:02}-{:02} {:02}:{:02}:{:02}.{:03}",
      tm.tm_year + 1900,
      tm.tm_mon + 1,
      tm.tm_mday,
      tm.tm_hour,
      tm.tm_min,
      tm.tm_sec,
      ms.count());
}

void *timer_increment(void *)
{
  while (true)
  {
    usleep(300000);
    sem_wait(shared_sem);
    shared_data->counter++;
    sem_post(shared_sem);
  }
  return nullptr;
}

void *input_listener(void *)
{
  while (true)
  {
    int newValue;
    std::cout << "Enter new counter value: ";
    if (std::cin >> newValue)
    {
      sem_wait(shared_sem);
      shared_data->counter = newValue;
      sem_post(shared_sem);
    }
    else
    {
      std::cin.clear();
      std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }
  }
  return nullptr;
}

void *spawner(void *)
{
  while (true)
  {
    sleep(3);

    if (waitpid(pid_copy_1, nullptr, WNOHANG) == 0 || waitpid(pid_copy_2, nullptr, WNOHANG) == 0)
    {
      logMessage("Both COPY_1 and COPY_2 are still running. Skipping spawn.");
      continue;
    }

    pid_copy_1 = fork();
    if (pid_copy_1 == 0)
    {
      logMessage("COPY_1 PID:" + std::to_string(getpid()) +
                 ". Timestamp: " + getCurrentTimestamp());

      sem_wait(shared_sem);
      shared_data->counter += 10;
      sem_post(shared_sem);

      logMessage("COPY_1 Exit timestamp: " + getCurrentTimestamp());
      exit(0);
    }

    pid_copy_2 = fork();
    if (pid_copy_2 == 0)
    {
      logMessage("COPY_2 PID:" + std::to_string(getpid()) +
                 ". Timestamp: " + getCurrentTimestamp());

      sem_wait(shared_sem);
      shared_data->counter *= 2;
      sem_post(shared_sem);

      sleep(2);

      sem_wait(shared_sem);
      shared_data->counter /= 2;
      sem_post(shared_sem);
      logMessage("COPY_2 Exit timestamp: " + getCurrentTimestamp());
      exit(0);
    }
  }

  return nullptr;
}

void *timer_logger(void *)
{
  while (true)
  {
    sleep(1);

    sem_wait(shared_sem);
    int temp = shared_data->counter;
    sem_post(shared_sem);

    logMessage("PID:" + std::to_string(getpid()) +
               ". Timestamp: " + getCurrentTimestamp() +
               ". Counter: " + std::to_string(temp));
  }
  return nullptr;
}

void cleanup()
{
  if (shared_data)
    munmap(shared_data, sizeof(SharedData));

  if (shared_sem)
    sem_close(shared_sem);

  if (log_fd.is_open())
    log_fd.close();
}

void sigint_handler(int)
{
  bool am_leader = false;

  sem_wait(shared_sem);
  am_leader = (shared_data->leader_pid == getpid());
  sem_post(shared_sem);

  cleanup();

  if (am_leader)
  {
    shm_unlink(shm_name);
    sem_unlink(sem_name);
    std::cout << "\nLeader cleaned shared resources.\n";
  }
  else
  {
    std::cout << "\nProcess exiting.\n";
  }

  _exit(0);
}
int main(int argc, char const *argv[])
{

  bool created = false;

  int shm_fd = shm_open(shm_name, O_CREAT | O_EXCL | O_RDWR, 0666);
  if (shm_fd >= 0)
  {
    created = true;
    ftruncate(shm_fd, sizeof(SharedData));
  }
  else
  {
    shm_fd = shm_open(shm_name, O_RDWR, 0666);
  }

  shared_data = (SharedData *)mmap(NULL, sizeof(SharedData),
                                   PROT_READ | PROT_WRITE,
                                   MAP_SHARED, shm_fd, 0);

  if (created)
  {
    shared_data->counter = 0;
    shared_data->leader_pid = 0;
  }

  shared_sem = sem_open(sem_name, O_CREAT, 0666, 1);
  sem_wait(shared_sem);

  if (shared_data->leader_pid == 0)
  {
    shared_data->leader_pid = getpid();
  }

  sem_post(shared_sem);
  log_fd.open(LOG_FILE, std::ios_base::app);
  signal(SIGINT, sigint_handler);

  pthread_t timerThread, inputThread;
  if (getpid() == shared_data->leader_pid)
  {
    pthread_t loggerThread, spawnerThread;
    logMessage("Main PID:" + std::to_string(getpid()) + ". Timestamp: " + getCurrentTimestamp());
    pthread_create(&loggerThread, nullptr, timer_logger, nullptr);
    pthread_create(&spawnerThread, nullptr, spawner, nullptr);
    pthread_create(&timerThread, nullptr, timer_increment, nullptr);
    pthread_create(&inputThread, nullptr, input_listener, nullptr);
    pthread_join(spawnerThread, nullptr);
    pthread_join(loggerThread, nullptr);
    pthread_join(timerThread, nullptr);
    pthread_join(inputThread, nullptr);
  }

  pthread_create(&timerThread, nullptr, timer_increment, nullptr);
  pthread_create(&inputThread, nullptr, input_listener, nullptr);

  pthread_join(timerThread, nullptr);
  pthread_join(inputThread, nullptr);
  return 0;
}