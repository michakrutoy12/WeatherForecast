new Vue({
    el: '#app',
    data: {
        currentWeather: null,
        weeklyForecast: [],
        unit: 'celsius', // celsius или fahrenheit
        loading: false,
        particlesCount: 30,
        particles: [],
        apiBaseUrl: 'http://localhost:8082/weather'
    },
    
    computed: {
        formattedDate() {
            const now = new Date();
            return now.toLocaleDateString('ru-RU', { 
                weekday: 'long', 
                year: 'numeric', 
                month: 'long', 
                day: 'numeric' 
            });
        },
        
        themeClass() {
            if (!this.currentWeather) return 'theme-default';
            
            const temp = this.currentWeather.temperature;
            
            if (temp < 0) return 'theme-cold';
            if (temp >= 0 && temp < 15) return 'theme-cool';
            if (temp >= 15 && temp < 25) return 'theme-warm';
            if (temp >= 25) return 'theme-hot';
            
            return 'theme-default';
        },
        
        averageTemp() {
            if (!this.weeklyForecast.length) return 0;
            const sum = this.weeklyForecast.reduce((acc, day) => acc + day.temperature, 0);
            return sum / this.weeklyForecast.length;
        },
        
        averageHumidity() {
            if (!this.weeklyForecast.length) return 0;
            const sum = this.weeklyForecast.reduce((acc, day) => acc + day.humidity, 0);
            return Math.round(sum / this.weeklyForecast.length);
        },
        
        maxTemp() {
            if (!this.weeklyForecast.length) return 0;
            return Math.max(...this.weeklyForecast.map(day => day.temperature));
        },
        
        minTemp() {
            if (!this.weeklyForecast.length) return 0;
            return Math.min(...this.weeklyForecast.map(day => day.temperature));
        }
    },
    
    mounted() {
        this.initParticles();
        this.loadWeatherData();
        // Обновляем данные каждые 5 минут
        setInterval(() => {
            this.loadWeatherData();
        }, 5 * 60 * 1000);
    },
    
    methods: {
        initParticles() {
            for (let i = 0; i < this.particlesCount; i++) {
                this.particles.push({
                    left: Math.random() * 100,
                    top: Math.random() * 100,
                    delay: Math.random() * 5,
                    duration: 3 + Math.random() * 4,
                    size: 2 + Math.random() * 6
                });
            }
        },
        
        getParticleStyle(index) {
            const p = this.particles[index];
            if (!p) return {};
            return {
                left: p.left + '%',
                top: p.top + '%',
                animationDelay: p.delay + 's',
                animationDuration: p.duration + 's',
                width: p.size + 'px',
                height: p.size + 'px'
            };
        },
        
        async loadWeatherData() {
            this.loading = true;
            try {
                // Загружаем все данные о погоде для Красноярска
                const response = await fetch(`${this.apiBaseUrl}`);
                console.log(response)
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                
                const data = await response.json();
                console.log(data)
                
                
                // Прогноз на неделю - все записи
                this.currentWeather = data[0];
                this.weeklyForecast = data
                console.log(this.currentWeather)
            } catch (error) {
                console.error('Ошибка загрузки данных:', error);
                alert('Не удалось загрузить данные о погоде. Проверьте подключение к API.');
                
                // Демо-данные для тестирования, если API недоступен
                this.loadDemoData();
            } finally {
                this.loading = false;
            }
        },
        
        loadDemoData() {
            console.log('Загрузка демо-данных');
            const demoData = [];
            const today = new Date();
            
            for (let i = 0; i < 7; i++) {
                const date = new Date(today);
                date.setDate(today.getDate() + i);
                
                demoData.push({
                    id: i + 1,
                    forecast_date: date.toISOString().split('T')[0],
                    temperature: Math.round(Math.sin(i) * 15 + 5),
                    humidity: Math.round(Math.random() * 40 + 40),
                    created_at: new Date().toISOString()
                });
            }
            
            this.weeklyForecast = demoData;
            this.currentWeather = demoData[0];
        },
        
        formatTemperature(celsius) {
                return Math.round(celsius);
        },
        
        getWeatherIconByTemp(temperature) {
            if (temperature < -20) return 'fas fa-snowman';
            if (temperature < 0) return 'fas fa-snowflake';
            if (temperature < 10) return 'fas fa-cloud-rain';
            if (temperature < 20) return 'fas fa-cloud-sun';
            if (temperature < 30) return 'fas fa-sun';
            return 'fas fa-temperature-high';
        },
        
        getWeatherDescription(temperature) {
            if (temperature < -30) return 'Экстремальный холод ❄️';
            if (temperature < -20) return 'Сильный мороз 🥶';
            if (temperature < -10) return 'Морозно 🧊';
            if (temperature < 0) return 'Холодно ❄️';
            if (temperature < 10) return 'Прохладно 🌧️';
            if (temperature < 20) return 'Тепло ☀️';
            if (temperature < 30) return 'Жарко 🔥';
            return 'Очень жарко 🥵';
        },
        
        formatDayOfWeek(dateString) {
            const date = new Date(dateString);
            const days = ['ВС', 'ПН', 'ВТ', 'СР', 'ЧТ', 'ПТ', 'СБ'];
            return days[date.getDay()];
        },
        
        formatDateRu(dateString) {
            const date = new Date(dateString);
            return date.toLocaleDateString('ru-RU', { 
                day: 'numeric', 
                month: 'numeric', 
                
            });
        }
    },
    
});