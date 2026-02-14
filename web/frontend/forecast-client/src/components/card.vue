<script setup>



</script>


<template>

    <body class="mainBody">

        <div class="leftSide">
            <div class="leftSide-upper">
                <h2>Day</h2>
                <h6>dd.mmmm.yyyy</h6>
                <h5>city</h5>
            </div>

            <div>
                <span>osadki.mark</span>
                <h1>Temperature</h1>
                <h2>osadki.name</h2>
            </div>

        </div>

        <div class="rightSide">

            <div>
                <h1>Vlaga</h1>
                <h1>osadki</h1>
                <h1>wind</h1>
            </div>

            <div>
                <div id="dayCard">
                    day
                </div>
            </div>



        </div>

    </body>

</template>

<style>

.mainBody{
    display: flex;
    font-size: 3vh;
    justify-content: center;
    vertical-align: middle;
    align-items: center;
}

.leftSide{
    background: linear-gradient(180deg, rgba(144, 129, 253, 1), rgba(216, 199, 255, 1));
-webkit-background: linear-gradient(180deg, rgba(144, 129, 253, 1), rgba(216, 199, 255, 1));
-moz-background: linear-gradient(180deg, rgba(144, 129, 253, 1), rgba(216, 199, 255, 1));
padding: 8px;
display: flex;
flex-direction: column;
text-align: left;
border-radius: 0.5rem;
justify-content: space-between;
}

.rightSide{
    background: #000080;
    padding: 8px;
    display: flex;
    align-content: center;
    flex-direction: column;
    text-align: justify;
    justify-content: space-between;
    box-shadow: 2px 2px 2px 2px, white inset;
    border-radius: 0px 0.5rem 0.5rem 0;
    height: 80%;
}
</style>