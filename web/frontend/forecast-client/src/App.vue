<script setup>
import { ref, computed } from 'vue'
import card from './components/card.vue'
import Card from './components/card.vue'
const mouseX = ref(50)
const mouseY = ref(50)

const handleMouseMove = (e) => {
  mouseX.value = (e.clientX / window.innerWidth) * 100
  mouseY.value = (e.clientY / window.innerHeight) * 100
}

const resetPosition = () => {
  mouseX.value = 50
  mouseY.value = 50
}


const bgStyle = computed(() => ({
  background: `radial-gradient(circle at ${mouseX.value}% ${mouseY.value}%, rgba(79, 56, 255, 1), rgba(56, 50, 103, 1), rgba(6, 0, 15, 1))`
}))
</script>

<template>

<body :style="bgStyle" class="base" @mousemove="handleMouseMove" @mouseleave="resetPosition">

  <main >


    <Card></Card>

  </main>


</body>

</template>

<style scoped>

.base{
  
  background-attachment: fixed;
  background-repeat: no-repeat;
  
  display:flex;
  flex-direction: column;
  align-items: center;
  text-align: center;
  vertical-align:middle;
  min-height: 100vh;
  padding: 0px;
  margin: 0px;
  justify-content: center;
  transition: 0.1s ease;
}

main{
  display: flex;
  flex-direction: column;
  background:#0B0B3A ;
  border-radius: 1rem;
  color: white;
  padding: 12px;
  box-shadow: 0px 4px 8px 8px rgba(0, 0, 0, 0.3);
}
</style>
