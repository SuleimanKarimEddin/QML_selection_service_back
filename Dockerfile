FROM php:8.2
RUN apt-get update -y \
    && apt-get install -y openssl zip unzip git libicu-dev zlib1g-dev

# Install PHP extensions
RUN docker-php-ext-install pdo 
RUN  docker-php-ext-configure intl 
RUN  docker-php-ext-install intl 

RUN apt-get install -y \
        libzip-dev \
        zip \
  && docker-php-ext-install zip

RUN curl -sS https://getcomposer.org/installer | php -- --install-dir=/usr/local/bin --filename=composer

WORKDIR /app
COPY . /app
RUN composer install

CMD php artisan serve --host=0.0.0.0 --port=8181
EXPOSE 8181
