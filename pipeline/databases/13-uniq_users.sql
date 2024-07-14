-- Create table
CREATE TABLE
    IF NOT EXISTS users (
        id INT NOT Null AUTO_INCREMENT PRIMARY KEY,
        email VARCHAR(255) NOT NULL UNIQUE,
        name VARCHAR(255)
    );