base_headers <- readLines('headers.txt', n=1)
cleaned_header <- gsub("\\s*\\([^()]*\\)", "", base_headers)

cleaned_header <- gsub("\\s+", " ", cleaned_header)
cleaned_header <- trimws(cleaned_header)
cleaned_header <- unlist(strsplit(cleaned_header, " "))
length(cleaned_header)