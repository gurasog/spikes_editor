# Kill processes on ports 10125-10129
kill -9 $(lsof -t -i:10125)
kill -9 $(lsof -t -i:10126)
kill -9 $(lsof -t -i:10127)
kill -9 $(lsof -t -i:10128)
kill -9 $(lsof -t -i:10129)