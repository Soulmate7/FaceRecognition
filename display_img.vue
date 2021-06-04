<template>
  <el-row>
    <el-col :span="12">
      <div class="grid-content bg-purple">
        <p>动态绑定路径_原图</p>
        <img :src="origin_image" alt="图片加载失败" title="assets中的图片">
      </div>
    </el-col>
    <el-col :span="12">
      <div class="grid-content bg-purple-light">
        <p>动态绑定路径_特征点图</p>
        <img :src="feature_image" alt="图片加载失败" title="assets中的图片">
        <router-view/>
      </div>
    </el-col>
  </el-row>
</template>


<script>
  export default {
    name: "display_img",
    data() {
      return {
        origin_image: require('@/assets/images/origin_image.jpg'),
        feature_image: require('@/assets/images/feature_image.jpg')
      }
    }
  }
</script>


<style>

</style>

